"""Scan the rose subset, audit labels/duplicates, and build a grouped split.

Related images (augmented copies of the same photo, exact or near duplicates)
are merged into one group, and whole groups are assigned to train/val/test
(~70/15/15, stratified by label). This prevents leakage between splits.

Example:
    python prepare_split.py --data-root /path/to/dataset --subset-keyword rose \
        --out splits/rose_split.json
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

from config import CLASSES, CLASS_KEYWORDS, IMG_EXTS

# Strips common augmentation / copy suffixes from file stems, e.g.
# "IMG_0012_aug3", "leaf_rot90", "photo_flip", "img (2)", "img_copy".
DEFAULT_AUG_REGEX = (
    r"([_\-\s]*(aug(mented)?|rot(ate)?|flip(ped)?|hflip|vflip|bright(ness)?|"
    r"contrast|zoom|shift|crop|noise|blur|copy)[_\-]?\d*)+$|\s*\(\d+\)$"
)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def match_label(rel_path, class_keywords):
    """Return the class whose keyword matches the deepest directory component."""
    parts = [p.lower() for p in rel_path.split(os.sep)[:-1]]
    for part in reversed(parts):
        hits = {c for c, kws in class_keywords.items() if any(k in part for k in kws)}
        if len(hits) == 1:
            return hits.pop()
        if len(hits) > 1:
            return None  # ambiguous folder name
    return None


def scan_images(root, subset_keyword, class_keywords):
    records, skipped = [], []
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in IMG_EXTS:
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if subset_keyword and subset_keyword.lower() not in rel.lower():
                continue
            label = match_label(rel, class_keywords)
            if label is None:
                skipped.append(rel)
                continue
            records.append({"path": rel, "label_name": label})
    return records, skipped


def stem_key(rel_path, aug_regex):
    stem = os.path.splitext(os.path.basename(rel_path))[0].lower()
    base = re.sub(aug_regex, "", stem) or stem
    return os.path.join(os.path.dirname(rel_path).lower(), base)


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path, size=8):
    """64-bit difference hash; robust to resizing and mild compression."""
    with Image.open(path) as img:
        g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def _popcount(x):
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def near_duplicate_pairs(hashes, threshold, chunk=256):
    h = np.array(hashes, dtype=np.uint64)
    pairs = []
    for start in range(0, len(h), chunk):
        block = h[start:start + chunk]
        for i, hi in enumerate(block):
            gi = start + i
            dist = _popcount(np.bitwise_xor(h[gi + 1:], hi))
            for j in np.nonzero(dist <= threshold)[0]:
                pairs.append((gi, gi + 1 + int(j)))
    return pairs


def build_groups(records, root, aug_regex, hash_threshold, use_filename_groups):
    n = len(records)
    uf = UnionFind(n)
    audit = {"exact_duplicate_pairs": 0, "near_duplicate_pairs": 0}

    if use_filename_groups:
        by_key = defaultdict(list)
        for i, r in enumerate(records):
            by_key[stem_key(r["path"], aug_regex)].append(i)
        for idx in by_key.values():
            for j in idx[1:]:
                uf.union(idx[0], j)

    by_md5 = defaultdict(list)
    hashes = []
    for i, r in enumerate(records):
        p = os.path.join(root, r["path"])
        by_md5[file_md5(p)].append(i)
        hashes.append(dhash(p) if hash_threshold >= 0 else 0)
    for idx in by_md5.values():
        audit["exact_duplicate_pairs"] += len(idx) - 1
        for j in idx[1:]:
            uf.union(idx[0], j)

    if hash_threshold >= 0:
        pairs = near_duplicate_pairs(hashes, hash_threshold)
        audit["near_duplicate_pairs"] = len(pairs)
        for a, b in pairs:
            uf.union(a, b)

    roots = {}
    group_ids = [roots.setdefault(uf.find(i), len(roots)) for i in range(n)]

    # Groups whose members carry different labels point to label noise.
    labels_per_group = defaultdict(set)
    members = defaultdict(list)
    for i, g in enumerate(group_ids):
        labels_per_group[g].add(records[i]["label_name"])
        members[g].append(records[i]["path"])
    conflicts = [
        {"group": g, "labels": sorted(ls), "paths": members[g][:20]}
        for g, ls in labels_per_group.items() if len(ls) > 1
    ]
    audit["num_groups"] = len(roots)
    audit["label_conflict_groups"] = conflicts
    return group_ids, audit


def stratified_group_split(records, group_ids, ratios, seed):
    """Assign whole groups to splits, stratified by each group's majority label."""
    groups = defaultdict(list)
    for i, g in enumerate(group_ids):
        groups[g].append(i)
    by_label = defaultdict(list)
    for g, idx in groups.items():
        majority = Counter(records[i]["label_name"] for i in idx).most_common(1)[0][0]
        by_label[majority].append(g)

    rng = random.Random(seed)
    assignment = {}
    names = ["train", "val", "test"]
    for label in sorted(by_label):
        gs = sorted(by_label[label])
        rng.shuffle(gs)
        total = sum(len(groups[g]) for g in gs)
        bounds = np.cumsum(ratios) * total
        count = 0
        for g in gs:
            # Place the group by the position of its midpoint in the cumulative count.
            mid = count + len(groups[g]) / 2
            assignment[g] = names[min(int(np.searchsorted(bounds, mid, side="right")), 2)]
            count += len(groups[g])
    return [assignment[g] for g in group_ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--subset-keyword", default="rose",
                    help="Only keep images whose relative path contains this text ('' = all).")
    ap.add_argument("--class-map", default=None,
                    help="JSON file {class_name: [folder keywords]} overriding config.CLASS_KEYWORDS.")
    ap.add_argument("--aug-regex", default=DEFAULT_AUG_REGEX)
    ap.add_argument("--no-filename-groups", action="store_true",
                    help="Do not group files by their stem with augmentation suffixes removed.")
    ap.add_argument("--hash-threshold", type=int, default=4,
                    help="Max dHash Hamming distance for near duplicates (-1 disables).")
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="splits/rose_split.json")
    args = ap.parse_args()

    class_keywords = CLASS_KEYWORDS
    if args.class_map:
        with open(args.class_map) as f:
            class_keywords = {k: [s.lower() for s in v] for k, v in json.load(f).items()}
    assert set(class_keywords) == set(CLASSES), f"class map must define exactly {CLASSES}"

    root = os.path.abspath(args.data_root)
    records, skipped = scan_images(root, args.subset_keyword, class_keywords)
    if not records:
        raise SystemExit("No labeled images found; check --data-root/--subset-keyword/--class-map.")
    print(f"Found {len(records)} labeled images ({len(skipped)} skipped, no/ambiguous label).")

    group_ids, audit = build_groups(records, root, args.aug_regex, args.hash_threshold,
                                    not args.no_filename_groups)
    ratios = np.array(args.ratios) / sum(args.ratios)
    splits = stratified_group_split(records, group_ids, ratios, args.seed)

    items = [
        {"path": r["path"], "label": CLASSES.index(r["label_name"]), "group": g, "split": s}
        for r, g, s in zip(records, group_ids, splits)
    ]
    counts = {s: Counter(CLASSES[it["label"]] for it in items if it["split"] == s)
              for s in ("train", "val", "test")}
    audit["skipped_paths"] = skipped[:200]
    audit["num_skipped"] = len(skipped)
    audit["counts"] = {s: dict(c) for s, c in counts.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"root": root, "classes": CLASSES, "seed": args.seed,
                   "ratios": ratios.tolist(), "items": items, "audit": audit}, f, indent=1)

    print(f"Groups: {audit['num_groups']}  exact dup pairs: {audit['exact_duplicate_pairs']}  "
          f"near dup pairs: {audit['near_duplicate_pairs']}  "
          f"label-conflict groups: {len(audit['label_conflict_groups'])}")
    for s, c in counts.items():
        print(f"  {s:5s}: {sum(c.values()):5d}  " + "  ".join(f"{k}={c.get(k, 0)}" for k in CLASSES))
    if audit["label_conflict_groups"]:
        print("WARNING: some groups mix labels; review 'label_conflict_groups' in the split file.")
    print(f"Saved split to {args.out}")


if __name__ == "__main__":
    main()
