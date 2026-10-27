"""MobileNetV2 backbone with a new 4-class head, plus freezing helpers."""
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def build_model(num_classes, pretrained=True, dropout=0.2):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights, width_mult=1.0)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes))
    return model


def set_trainable(model, stage, unfreeze_from):
    """Stage 1: only the head trains. Stage 2: also features[unfreeze_from:]."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if stage == 2:
        for block in model.features[unfreeze_from:]:
            for p in block.parameters():
                p.requires_grad = True


def set_bn_mode(model, stage, unfreeze_from, freeze_all_bn):
    """Call after model.train(): keep BatchNorm running stats fixed where required.

    Frozen blocks always keep their statistics; with freeze_all_bn the unfrozen
    blocks do too (recommended for small batches).
    """
    for i, block in enumerate(model.features):
        if stage == 1 or i < unfreeze_from or freeze_all_bn:
            for m in block.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


def param_groups(model, unfreeze_from, lr_backbone, lr_head):
    backbone = [p for p in model.features[unfreeze_from:].parameters() if p.requires_grad]
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    return [{"params": backbone, "lr": lr_backbone}, {"params": head, "lr": lr_head}]


def save_checkpoint(path, model, meta):
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
