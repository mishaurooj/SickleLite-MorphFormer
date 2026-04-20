"""
SickleLite-MorphFormer: Updated Full Implementation
==================================================
Adds:
1. Main benchmark / baseline comparison block (separate from ablations)
2. 6 ablations already requested
3. Unified results tables and summary figures
4. Benchmark comparison figure for all trained models
5. Modes: train | benchmark | ablation | all | eval

Dataset layout expected at DATA_ROOT:
    DATA_ROOT/
        Positive/
            Labelled/
            Unlabelled/
        Negative/

Example usage:
    python sickle_morph_updated.py --data_root "D:/sickle/dataset" --mode benchmark
    python sickle_morph_updated.py --data_root "D:/sickle/dataset" --mode ablation
    python sickle_morph_updated.py --data_root "D:/sickle/dataset" --mode all
"""

import os
import sys
import time
import random
import logging
import argparse
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, brier_score_loss,
)

warnings.filterwarnings("ignore")


class Config:
    DATA_ROOT = r"D:\sickle\dataset"
    OUTPUT_DIR = "outputs"

    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    PATIENCE = 10
    NUM_WORKERS = 8
    GRAD_ACCUM = 2
    AMP = True
    SEED = 42

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    LAMBDA_CONS = 0.3
    LAMBDA_ATTN = 0.2

    KFOLDS = 5

    CNN_CHANNELS = 64
    TRANS_DIM = 128
    TRANS_BLOCKS = 3
    NUM_CLASSES = 2

    BENCHMARK_QUICK_EPOCHS = 15
    ABLATION_QUICK_EPOCHS = 20

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CFG = Config()


BENCHMARK_MODELS = [
    "ResNet18",
    "ResNet50",
    "DenseNet121",
    "EfficientNet-B0",
    "MobileNetV3-Small",
    "ShuffleNetV2-0.5",
    "ConvNeXt-Tiny",
    "MobileViT-XS",
    "Swin-Tiny",
    "CoAtNet-0",
    "SickleLite-MorphFormer",
]

METRIC_COLS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
    "mcc", "kappa", "auroc", "auprc", "brier", "ece",
    "params_M", "flops_G", "epoch_time_s", "total_train_min", "infer_ms", "throughput_img_s"
]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


logger = get_logger("sickle", os.path.join(CFG.OUTPUT_DIR, "logs", "training.log"))


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(data_root: str) -> Tuple[List[str], List[int], List[str]]:
    paths, labels, tags = [], [], []
    root = Path(data_root)

    neg_dir = root / "Negative"
    for p in sorted(neg_dir.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            paths.append(str(p))
            labels.append(0)
            tags.append("neg")

    pos_lab = root / "Positive" / "Labelled"
    for p in sorted(pos_lab.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            paths.append(str(p))
            labels.append(1)
            tags.append("pos_lab")

    pos_unlab = root / "Positive" / "Unlabelled"
    for p in sorted(pos_unlab.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            paths.append(str(p))
            labels.append(1)
            tags.append("pos_unlab")

    logger.info(
        f"Dataset loaded: {labels.count(0)} negative | {labels.count(1)} positive "
        f"({tags.count('pos_lab')} labelled + {tags.count('pos_unlab')} unlabelled)"
    )
    return paths, labels, tags


def stratified_split(paths, labels, tags, seed=42):
    idx = list(range(len(paths)))
    idx_trval, idx_test = train_test_split(
        idx, test_size=CFG.TEST_RATIO, stratify=labels, random_state=seed
    )
    val_frac = CFG.VAL_RATIO / (CFG.TRAIN_RATIO + CFG.VAL_RATIO)
    labels_trval = [labels[i] for i in idx_trval]
    idx_tr, idx_val = train_test_split(
        idx_trval, test_size=val_frac, stratify=labels_trval, random_state=seed
    )

    def pick(idxs):
        return ([paths[i] for i in idxs], [labels[i] for i in idxs], [tags[i] for i in idxs])

    return pick(idx_tr), pick(idx_val), pick(idx_test)


def make_train_aug(is_minority: bool) -> A.Compose:
    if is_minority:
        return A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            A.Rotate(limit=20, p=0.7),
            A.Affine(scale=(0.85, 1.15), translate_percent=0.15, shear=15, p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(5, 9), p=0.5),
            A.ElasticTransform(alpha=120, sigma=10, p=0.6),
            A.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=10, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussianBlur(blur_limit=(5, 9), p=0.3),
        A.Normalize(mean=CFG.MEAN, std=CFG.STD),
        ToTensorV2(),
    ])


def make_val_aug() -> A.Compose:
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=CFG.MEAN, std=CFG.STD),
        ToTensorV2(),
    ])


def make_consistency_aug() -> A.Compose:
    return A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.Normalize(mean=CFG.MEAN, std=CFG.STD),
        ToTensorV2(),
    ])


class SickleDataset(Dataset):
    def __init__(self, paths, labels, tags, mode="train"):
        self.paths = paths
        self.labels = labels
        self.tags = tags
        self.mode = mode
        self.min_aug = make_train_aug(True)
        self.maj_aug = make_train_aug(False)
        self.eval_aug = make_val_aug()
        self.cons_aug = make_consistency_aug()

    def __len__(self):
        return len(self.paths)

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            return np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        img = self._load(self.paths[idx])
        lbl = self.labels[idx]
        tag = self.tags[idx]
        if self.mode == "train":
            aug = self.min_aug if lbl == 1 else self.maj_aug
            x1 = aug(image=img)["image"]
            x2 = self.cons_aug(image=img)["image"] if lbl == 1 else x1.clone()
            is_labelled = torch.tensor(1 if tag == "pos_lab" else 0, dtype=torch.float32)
            return x1, x2, torch.tensor(lbl, dtype=torch.long), is_labelled
        x = self.eval_aug(image=img)["image"]
        return x, torch.tensor(lbl, dtype=torch.long)


def make_weighted_sampler(labels):
    counts = np.bincount(labels)
    w = 1.0 / counts
    sample_w = np.array([w[l] for l in labels])
    return WeightedRandomSampler(sample_w, len(labels), replacement=True)


def get_loaders(tr_data, val_data, te_data):
    tr_set = SickleDataset(*tr_data, mode="train")
    va_set = SickleDataset(*val_data, mode="val")
    te_set = SickleDataset(*te_data, mode="test")
    sampler = make_weighted_sampler(tr_data[1])
    tr_loader = DataLoader(tr_set, batch_size=CFG.BATCH_SIZE, sampler=sampler,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)
    va_loader = DataLoader(va_set, batch_size=CFG.BATCH_SIZE, shuffle=False,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)
    te_loader = DataLoader(te_set, batch_size=CFG.BATCH_SIZE, shuffle=False,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)
    return tr_loader, va_loader, te_loader


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu6(self.bn(self.pw(self.dw(x))))


class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, expand=4):
        super().__init__()
        mid = in_c * expand
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            nn.Conv2d(mid, out_c, 1, bias=False), nn.BatchNorm2d(out_c),
        )
        self.skip = (in_c == out_c)

    def forward(self, x):
        out = self.block(x)
        return out + x if self.skip else out


class FastStem(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            DepthwiseSeparable(32, 32),
            DepthwiseSeparable(32, 64),
            InvertedResidual(64, 64),
            InvertedResidual(64, out_channels),
        )

    def forward(self, x):
        return self.stem(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        return torch.sigmoid(self.fc(self.avg(x)) + self.fc(self.max(x)))


class SpatialGating(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DSRM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        branch_ch = max(channels // 5, 8)
        actual = branch_ch * 5
        self.b13 = nn.Sequential(nn.Conv2d(channels, branch_ch, (1, 3), padding=(0, 1), bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU6(inplace=True))
        self.b31 = nn.Sequential(nn.Conv2d(channels, branch_ch, (3, 1), padding=(1, 0), bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU6(inplace=True))
        self.b15 = nn.Sequential(nn.Conv2d(channels, branch_ch, (1, 5), padding=(0, 2), bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU6(inplace=True))
        self.b51 = nn.Sequential(nn.Conv2d(channels, branch_ch, (5, 1), padding=(2, 0), bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU6(inplace=True))
        self.bd = nn.Sequential(nn.Conv2d(channels, branch_ch, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(branch_ch), nn.ReLU6(inplace=True))
        self.fuse = nn.Conv2d(actual, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)
        self.sg = SpatialGating()

    def forward(self, x):
        cats = torch.cat([self.b13(x), self.b31(x), self.b15(x), self.b51(x), self.bd(x)], dim=1)
        out = self.bn(self.fuse(cats))
        out = out * self.ca(out) * self.sg(out)
        return out + x


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, eps=1e-6):
        super().__init__()
        self.heads = heads
        self.eps = eps
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        H = self.heads
        qkv = self.qkv(x).reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1 + self.eps
        k = F.elu(k) + 1 + self.eps
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        denom = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2))
        out = torch.einsum("bhnd,bhde->bhne", q, kv) / (denom.unsqueeze(-1) + self.eps)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LCTB(nn.Module):
    def __init__(self, in_channels=64, token_dim=128, n_blocks=3):
        super().__init__()
        self.patch_proj = nn.Conv2d(in_channels, token_dim, 4, stride=4, bias=False)
        self.norm = nn.BatchNorm2d(token_dim)
        self.blocks = nn.ModuleList([TransformerBlock(token_dim) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        tokens = F.relu6(self.norm(self.patch_proj(x)))
        B, D, h, w = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.pool(tokens.transpose(1, 2)).squeeze(-1)


class GMCF(nn.Module):
    def __init__(self, cnn_dim, trans_dim, out_dim):
        super().__init__()
        total = cnn_dim + cnn_dim + trans_dim
        self.gate = nn.Sequential(
            nn.Linear(total, max(total // 2, 16)), nn.ReLU(inplace=True),
            nn.Linear(max(total // 2, 16), total), nn.Sigmoid(),
        )
        self.proj = nn.Linear(total, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, f_cnn, f_shape, f_ctx):
        cat = torch.cat([f_cnn, f_shape, f_ctx], dim=-1)
        gate = self.gate(cat)
        return self.norm(self.proj(cat * gate))


class SickleLiteMorphFormer(nn.Module):
    def __init__(self, cnn_channels=64, trans_dim=128, trans_blocks=3, num_classes=2, fusion_dim=256):
        super().__init__()
        self.stem = FastStem(cnn_channels)
        self.dsrm = DSRM(cnn_channels)
        self.lctb = LCTB(cnn_channels, trans_dim, trans_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = GMCF(cnn_channels, trans_dim, fusion_dim)
        self.drop = nn.Dropout(0.3)
        self.cls_head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
        self.morph_proj = nn.Sequential(nn.Linear(fusion_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 32))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        cnn_feat = self.stem(x)
        shape_feat = self.dsrm(cnn_feat)
        ctx = self.lctb(cnn_feat)
        f_cnn = self.pool(cnn_feat).flatten(1)
        f_shape = self.pool(shape_feat).flatten(1)
        fused = self.drop(self.fusion(f_cnn, f_shape, ctx))
        return fused

    def forward(self, x):
        fused = self.encode(x)
        return self.cls_head(fused), self.morph_proj(fused)

    def predict(self, x):
        return self.forward(x)[0]


class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, lam_cons=0.3, lam_attn=0.2, label_smoothing=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.lam_cons = lam_cons
        self.lam_attn = lam_attn

    def consistency_loss(self, z1, z2, labels):
        mask = (labels == 1).float().unsqueeze(1)
        if mask.sum() == 0:
            return z1.new_tensor(0.0)
        diff = F.mse_loss(z1, z2.detach(), reduction="none")
        return (diff * mask).sum() / (mask.sum() + 1e-8)

    def attention_loss(self, morph, is_labelled, labels):
        mask = ((labels == 1) & (is_labelled == 1)).float().unsqueeze(1)
        if mask.sum() == 0:
            return morph.new_tensor(0.0)
        selected = morph * mask
        return -selected.var(dim=0).mean()

    def forward(self, logits, labels, z1, z2, is_labelled):
        l_cls = self.ce(logits, labels)
        l_cons = self.consistency_loss(z1, z2, labels)
        l_attn = self.attention_loss(z1, is_labelled, labels)
        total = l_cls + self.lam_cons * l_cons + self.lam_attn * l_attn
        return total, l_cls, l_cons, l_attn


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    m = {}
    m["accuracy"] = accuracy_score(y_true, y_pred)
    m["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    m["precision"] = precision_score(y_true, y_pred, zero_division=0)
    m["recall"] = recall_score(y_true, y_pred, zero_division=0)
    m["specificity"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    m["f1"] = f1_score(y_true, y_pred, zero_division=0)
    m["mcc"] = matthews_corrcoef(y_true, y_pred)
    m["kappa"] = cohen_kappa_score(y_true, y_pred)
    if len(np.unique(y_true)) > 1:
        m["auroc"] = roc_auc_score(y_true, y_prob)
        m["auprc"] = average_precision_score(y_true, y_prob)
    else:
        m["auroc"] = float("nan")
        m["auprc"] = float("nan")
    m["brier"] = brier_score_loss(y_true, y_prob)
    return m


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, device):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE).to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs / 1e9
    except Exception:
        return float("nan")


class BaselineWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self._fake_morph_dim = 32

    def forward(self, x):
        logits = self.model(x)
        morph = logits.new_zeros(x.size(0), self._fake_morph_dim)
        return logits, morph

    def predict(self, x):
        return self.model(x)


def get_baseline_model(name: str, num_classes: int = 2) -> nn.Module:
    import timm
    backbone_map = {
        "ResNet18": "resnet18",
        "ResNet50": "resnet50",
        "DenseNet121": "densenet121",
        "EfficientNet-B0": "efficientnet_b0",
        "MobileNetV3-Small": "mobilenetv3_small_100",
        "ShuffleNetV2-0.5": "shufflenet_v2_x0_5",
        "ConvNeXt-Tiny": "convnext_tiny",
        "MobileViT-XS": "mobilevit_xs",
        "Swin-Tiny": "swin_tiny_patch4_window7_224",
        "CoAtNet-0": "coatnet_0_rw_224",
    }
    if name not in backbone_map:
        raise ValueError(f"Unknown baseline: {name}")
    return timm.create_model(backbone_map[name], pretrained=False, num_classes=num_classes)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps=2):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        x1, x2, labels, is_lab = [b.to(device) for b in batch]
        with autocast(enabled=CFG.AMP):
            logits, z1 = model(x1)
            _, z2 = model(x2)
            loss, *_ = criterion(logits, labels, z1, z2, is_lab)
            loss = loss / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    ce = nn.CrossEntropyLoss()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        if len(batch) == 4:
            x, _, labels, _ = [b.to(device) for b in batch]
        else:
            x, labels = batch[0].to(device), batch[1].to(device)
        logits, _ = model(x)
        total_loss += ce(logits, labels).item()
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["ece"] = expected_calibration_error(y_true, y_prob)
    return total_loss / len(loader), metrics, y_true, y_pred, y_prob


def measure_inference_latency(model, device, n=100, warmup=10):
    model.eval()
    dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model.predict(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            model.predict(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    return ms


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _, __, output):
        self.acts = output.detach()

    def _save_grads(self, _, grad_input, grad_output):
        self.grads = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        self.model.eval()
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x.requires_grad_(True)
        logits, _ = self.model(x)
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.acts).sum(dim=1).squeeze()
        cam = F.relu(cam).detach().cpu().numpy()
        cam = cv2.resize(cam, (CFG.IMG_SIZE, CFG.IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def save_training_curves(tr_loss, va_loss, tr_acc, va_f1, save_dir, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tr_loss, label="Train Loss")
    axes[0].plot(va_loss, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(tr_acc, label="Train Acc")
    axes[1].plot(va_f1, label="Val F1")
    axes[1].set_title("Accuracy / F1")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", f"{name}_curves.png"), dpi=150)
    plt.close()


def save_confusion_matrix(y_true, y_pred, save_dir, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Sickle"], yticklabels=["Normal", "Sickle"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", f"{name}_cm.png"), dpi=150)
    plt.close()


def save_roc_pr(y_true, y_prob, save_dir, name):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"AUROC={roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--k")
    axes[0].set_title("ROC")
    axes[0].legend()
    axes[0].grid(True)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    axes[1].plot(rec, prec, label=f"AUPRC={pr_auc:.4f}")
    axes[1].set_title("PR Curve")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", f"{name}_roc_pr.png"), dpi=150)
    plt.close()


def save_bar(df: pd.DataFrame, metric: str, save_dir: str, name: str):
    df = df.sort_values(metric, ascending=False)
    plt.figure(figsize=(max(9, len(df) * 1.0), 5))
    bars = plt.bar(df["variant"], df[metric])
    for b, v in zip(bars, df[metric]):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(f"{name} - {metric}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", f"{name}_{metric}.png"), dpi=150)
    plt.close()


def save_scatter_efficiency(df: pd.DataFrame, save_dir: str, name: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(df["params_M"], df["f1"])
    for _, r in df.iterrows():
        plt.text(r["params_M"], r["f1"], r["variant"], fontsize=8)
    plt.xlabel("Parameters (M)")
    plt.ylabel("F1")
    plt.title(f"{name} - F1 vs Parameters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", f"{name}_f1_vs_params.png"), dpi=150)
    plt.close()


def save_gradcam_grid(model, te_dataset, save_dir, name="gradcam_samples", n=8):
    if not hasattr(model, "stem"):
        logger.info("Grad-CAM skipped: baseline wrapper has no FastStem target layer.")
        return
    target_layer = model.stem.stem[-1].block[-1] if hasattr(model.stem.stem[-1], "block") else model.stem.stem[-1]
    gcam = GradCAM(model, target_layer)
    fig, axes = plt.subplots(2, max(1, n // 2), figsize=(16, 6))
    axes = np.array(axes).reshape(-1)
    indices = random.sample(range(len(te_dataset)), min(n, len(te_dataset)))
    for i, idx in enumerate(indices):
        img_t, label = te_dataset[idx]
        cam = gcam(img_t.to(CFG.DEVICE))
        mean = np.array(CFG.MEAN).reshape(3, 1, 1)
        std = np.array(CFG.STD).reshape(3, 1, 1)
        img_np = (img_t.numpy() * std + mean).clip(0, 1).transpose(1, 2, 0)
        axes[i].imshow(img_np)
        axes[i].imshow(cam, alpha=0.45, cmap="jet")
        axes[i].set_title("Sickle" if label.item() == 1 else "Normal", fontsize=9)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", f"{name}.png"), dpi=150)
    plt.close()


def train_full(model, tr_loader, va_loader, te_loader, run_name="main", num_epochs=None, patience=None, save_dir=None):
    num_epochs = num_epochs or CFG.NUM_EPOCHS
    patience = patience or CFG.PATIENCE
    save_dir = save_dir or CFG.OUTPUT_DIR
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    device = CFG.DEVICE

    all_labels = [lbl.item() for _, _, lbl, _ in tr_loader.dataset]
    counts = np.bincount(all_labels)
    cls_w = torch.tensor([1.0, counts[0] / counts[1]], dtype=torch.float32).to(device)
    criterion = CombinedLoss(cls_w, CFG.LAMBDA_CONS, CFG.LAMBDA_ATTN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = warmup_cosine_scheduler(optimizer, CFG.WARMUP_EPOCHS, num_epochs)
    scaler = GradScaler(enabled=CFG.AMP)

    best_f1 = -1.0
    best_path = os.path.join(save_dir, f"{run_name}_best.pth")
    no_improve = 0
    tr_losses, va_losses, tr_accs, va_f1s, epoch_times = [], [], [], [], []

    for epoch in range(num_epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, optimizer, scaler, device, CFG.GRAD_ACCUM)
        va_loss, va_m, *_ = evaluate(model, va_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        tr_accs.append(tr_acc)
        va_f1s.append(va_m["f1"])
        logger.info(f"[{run_name}] epoch {epoch+1:03d}/{num_epochs} | tr_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_f1={va_m['f1']:.4f} auroc={va_m.get('auroc',0):.4f} | {elapsed:.1f}s")
        if va_m["f1"] > best_f1:
            best_f1 = va_m["f1"]
            no_improve = 0
            torch.save({"epoch": epoch, "model": model.state_dict()}, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    _, te_m, y_true, y_pred, y_prob = evaluate(model, te_loader, device)
    te_m["params_M"] = count_params(model) / 1e6
    te_m["flops_G"] = count_flops(model, device)
    te_m["epoch_time_s"] = float(np.mean(epoch_times))
    te_m["total_train_min"] = float(np.sum(epoch_times) / 60.0)
    te_m["infer_ms"] = measure_inference_latency(model, device)
    te_m["throughput_img_s"] = 1000.0 / te_m["infer_ms"] if te_m["infer_ms"] > 0 else float("nan")

    save_training_curves(tr_losses, va_losses, tr_accs, va_f1s, save_dir, run_name)
    save_confusion_matrix(y_true, y_pred, save_dir, run_name)
    save_roc_pr(y_true, y_prob, save_dir, run_name)
    return te_m, model, (y_true, y_pred, y_prob)


def quick_variant_run(model, tr_loader, va_loader, te_loader, tag, save_dir, epochs):
    device = CFG.DEVICE
    model = model.to(device)
    all_labels = [lbl.item() for _, _, lbl, _ in tr_loader.dataset]
    counts = np.bincount(all_labels)
    cls_w = torch.tensor([1.0, counts[0] / counts[1]], dtype=torch.float32).to(device)
    criterion = CombinedLoss(cls_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = warmup_cosine_scheduler(optimizer, 3, epochs)
    scaler = GradScaler(enabled=CFG.AMP)
    best_f1 = -1.0
    best_state = deepcopy(model.state_dict())
    times = []
    no_imp = 0
    for ep in range(epochs):
        t0 = time.time()
        train_one_epoch(model, tr_loader, criterion, optimizer, scaler, device, CFG.GRAD_ACCUM)
        _, va_m, *_ = evaluate(model, va_loader, device)
        scheduler.step()
        times.append(time.time() - t0)
        if va_m["f1"] > best_f1:
            best_f1 = va_m["f1"]
            best_state = deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 5:
                break
    model.load_state_dict(best_state)
    _, te_m, y_true, y_pred, y_prob = evaluate(model, te_loader, device)
    te_m["params_M"] = count_params(model) / 1e6
    te_m["flops_G"] = count_flops(model, device)
    te_m["epoch_time_s"] = float(np.mean(times))
    te_m["total_train_min"] = float(np.sum(times) / 60.0)
    te_m["infer_ms"] = measure_inference_latency(model, device)
    te_m["throughput_img_s"] = 1000.0 / te_m["infer_ms"] if te_m["infer_ms"] > 0 else float("nan")
    te_m["variant"] = tag
    save_confusion_matrix(y_true, y_pred, save_dir, tag)
    save_roc_pr(y_true, y_prob, save_dir, tag)
    return te_m, model


class SEBlock(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        hidden = max(ch // ratio, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(ch, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, ch), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(self.pool(x).flatten(1)).view(x.size(0), x.size(1), 1, 1)
        return x * w


class CBAMBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sg = SpatialGating()

    def forward(self, x):
        return x * self.ca(x) * self.sg(x)


class MorphVariantModel(nn.Module):
    def __init__(self, morph_type="dsrm", num_classes=2):
        super().__init__()
        C, D, Fdim = CFG.CNN_CHANNELS, CFG.TRANS_DIM, 256
        self.stem = FastStem(C)
        self.lctb = LCTB(C, D, CFG.TRANS_BLOCKS)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fusion = GMCF(C, D, Fdim)
        self.drop = nn.Dropout(0.3)
        self.cls_head = nn.Sequential(nn.Linear(Fdim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
        self.morph_proj = nn.Sequential(nn.Linear(Fdim, 64), nn.ReLU(), nn.Linear(64, 32))
        if morph_type == "none":
            self.morph_mod = nn.Identity()
        elif morph_type == "se":
            self.morph_mod = SEBlock(C)
        elif morph_type == "cbam":
            self.morph_mod = CBAMBlock(C)
        elif morph_type == "deform":
            self.morph_mod = nn.Sequential(nn.Conv2d(C, C, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(C), nn.ReLU6(inplace=True))
        else:
            self.morph_mod = DSRM(C)

    def forward(self, x):
        feat = self.stem(x)
        shape = self.morph_mod(feat)
        ctx = self.lctb(feat)
        f_cnn = self.pool(feat).flatten(1)
        f_shape = self.pool(shape).flatten(1)
        fused = self.drop(self.fusion(f_cnn, f_shape, ctx))
        return self.cls_head(fused), self.morph_proj(fused)

    def predict(self, x):
        return self.forward(x)[0]


class ContextVariantModel(nn.Module):
    def __init__(self, ctx_type="lctb", num_classes=2):
        super().__init__()
        C, D, Fdim = CFG.CNN_CHANNELS, CFG.TRANS_DIM, 256
        self.stem = FastStem(C)
        self.dsrm = DSRM(C)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ctx_type = ctx_type
        if ctx_type == "none":
            self.ctx_mod = None
        else:
            n_blocks = 2 if ctx_type == "linear" else CFG.TRANS_BLOCKS
            self.ctx_mod = LCTB(C, D, n_blocks)
        self.fusion = GMCF(C, D if self.ctx_mod is not None else C, Fdim)
        self.drop = nn.Dropout(0.3)
        self.cls_head = nn.Sequential(nn.Linear(Fdim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
        self.morph_proj = nn.Sequential(nn.Linear(Fdim, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, x):
        feat = self.stem(x)
        shape = self.dsrm(feat)
        f_c = self.pool(feat).flatten(1)
        f_s = self.pool(shape).flatten(1)
        ctx = self.ctx_mod(feat) if self.ctx_mod is not None else f_c
        fused = self.drop(self.fusion(f_c, f_s, ctx))
        return self.cls_head(fused), self.morph_proj(fused)

    def predict(self, x):
        return self.forward(x)[0]


class FusionVariantModel(nn.Module):
    def __init__(self, fusion_type="gmcf", num_classes=2):
        super().__init__()
        C, D, Fdim = CFG.CNN_CHANNELS, CFG.TRANS_DIM, 256
        self.stem = FastStem(C)
        self.dsrm = DSRM(C)
        self.lctb = LCTB(C, D, CFG.TRANS_BLOCKS)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ftype = fusion_type
        total = C + C + D
        if fusion_type == "concat":
            self.fuse = nn.Sequential(nn.Linear(total, Fdim), nn.LayerNorm(Fdim))
        elif fusion_type == "sum":
            self.proj_c = nn.Linear(C, Fdim)
            self.proj_s = nn.Linear(C, Fdim)
            self.proj_t = nn.Linear(D, Fdim)
            self.fuse = None
        else:
            self.fuse = GMCF(C, D, Fdim)
        self.drop = nn.Dropout(0.3)
        self.cls_head = nn.Sequential(nn.Linear(Fdim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
        self.morph_proj = nn.Sequential(nn.Linear(Fdim, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, x):
        feat = self.stem(x)
        shape = self.dsrm(feat)
        ctx = self.lctb(feat)
        f_c = self.pool(feat).flatten(1)
        f_s = self.pool(shape).flatten(1)
        if self.ftype == "concat":
            fused = self.fuse(torch.cat([f_c, f_s, ctx], dim=-1))
        elif self.ftype == "sum":
            fused = self.proj_c(f_c) + self.proj_s(f_s) + self.proj_t(ctx)
        else:
            fused = self.fuse(f_c, f_s, ctx)
        fused = self.drop(fused)
        return self.cls_head(fused), self.morph_proj(fused)

    def predict(self, x):
        return self.forward(x)[0]


class LossVariantA5(nn.Module):
    def __init__(self, strategy, class_weights=None):
        super().__init__()
        self.strategy = strategy
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def forward(self, logits, labels, z1, z2, is_labelled):
        l_cls = self.ce(logits, labels)
        if self.strategy == "supervised":
            return l_cls, l_cls, l_cls.new_tensor(0.0), l_cls.new_tensor(0.0)
        mask = (labels == 1).float().unsqueeze(1)
        if mask.sum() > 0:
            l_cons = (F.mse_loss(z1, z2.detach(), reduction="none") * mask).sum() / (mask.sum() + 1e-8)
        else:
            l_cons = l_cls.new_tensor(0.0)
        if self.strategy == "consistency":
            return l_cls + 0.3 * l_cons, l_cls, l_cons, l_cls.new_tensor(0.0)
        lab_mask = ((labels == 1) & (is_labelled == 1)).float().mean()
        l_attn = lab_mask * l_cls
        return l_cls + 0.3 * l_cons + 0.2 * l_attn, l_cls, l_cons, l_attn


class SimpleFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def benchmark_comparison(tr_loader, va_loader, te_loader, save_dir):
    logger.info("=" * 70)
    logger.info("Main Benchmark Comparison Block")
    logger.info("=" * 70)
    rows = []
    for name in BENCHMARK_MODELS:
        logger.info(f"Benchmark model: {name}")
        try:
            if name == "SickleLite-MorphFormer":
                model = SickleLiteMorphFormer()
            else:
                model = BaselineWrapper(get_baseline_model(name))
            result, model = quick_variant_run(model, tr_loader, va_loader, te_loader, name, save_dir, CFG.BENCHMARK_QUICK_EPOCHS)
            rows.append(result)
            if name == "SickleLite-MorphFormer":
                save_gradcam_grid(model, te_loader.dataset, save_dir, name="benchmark_gradcam_proposed")
        except Exception as e:
            logger.warning(f"Skipping {name}: {e}")
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "benchmark_main_comparison.csv")
    df.to_csv(csv_path, index=False)
    save_bar(df, "f1", save_dir, "benchmark_main")
    save_bar(df, "accuracy", save_dir, "benchmark_main")
    save_bar(df, "infer_ms", save_dir, "benchmark_latency")
    save_scatter_efficiency(df, save_dir, "benchmark_main")
    logger.info(f"Benchmark saved to {csv_path}")
    return df


def ablation_A1_backbone(tr_loader, va_loader, te_loader, save_dir):
    variants = ["MobileNetV3-Small", "EfficientNet-B0", "ShuffleNetV2-0.5", "Swin-Tiny", "ConvNeXt-Tiny", "SickleLite-MorphFormer"]
    rows = []
    for v in variants:
        model = SickleLiteMorphFormer() if v == "SickleLite-MorphFormer" else BaselineWrapper(get_baseline_model(v))
        result, _ = quick_variant_run(model, tr_loader, va_loader, te_loader, f"A1_{v}", save_dir, CFG.ABLATION_QUICK_EPOCHS)
        result["variant"] = v
        rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "ablation_A1_backbone.csv"), index=False)
    save_bar(df, "f1", save_dir, "A1_backbone")
    return df


def ablation_A2_dsrm(tr_loader, va_loader, te_loader, save_dir):
    rows = []
    for v in ["none", "se", "cbam", "deform", "dsrm", "full"]:
        mt = "dsrm" if v == "full" else v
        model = MorphVariantModel(morph_type=mt)
        result, _ = quick_variant_run(model, tr_loader, va_loader, te_loader, f"A2_{v}", save_dir, CFG.ABLATION_QUICK_EPOCHS)
        result["variant"] = v
        rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "ablation_A2_dsrm.csv"), index=False)
    save_bar(df, "f1", save_dir, "A2_dsrm")
    return df


def ablation_A3_context(tr_loader, va_loader, te_loader, save_dir):
    rows = []
    for v in ["none", "mhsa", "linear", "lctb"]:
        model = ContextVariantModel(ctx_type=v)
        result, _ = quick_variant_run(model, tr_loader, va_loader, te_loader, f"A3_{v}", save_dir, CFG.ABLATION_QUICK_EPOCHS)
        result["variant"] = v
        rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "ablation_A3_context.csv"), index=False)
    save_bar(df, "f1", save_dir, "A3_context")
    return df


def ablation_A4_fusion(tr_loader, va_loader, te_loader, save_dir):
    rows = []
    for v in ["concat", "sum", "gate", "gmcf"]:
        model = FusionVariantModel(fusion_type=v)
        result, _ = quick_variant_run(model, tr_loader, va_loader, te_loader, f"A4_{v}", save_dir, CFG.ABLATION_QUICK_EPOCHS)
        result["variant"] = v
        rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "ablation_A4_fusion.csv"), index=False)
    save_bar(df, "f1", save_dir, "A4_fusion")
    return df


def run_custom_loss_experiment(model, tr_loader, va_loader, te_loader, criterion, save_dir, tag, epochs):
    model = model.to(CFG.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sch = warmup_cosine_scheduler(opt, 3, epochs)
    scaler = GradScaler(enabled=CFG.AMP)
    best_f1 = -1.0
    best_state = deepcopy(model.state_dict())
    times = []
    no_imp = 0
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        for x1, x2, lbls, is_lab in tr_loader:
            x1, x2, lbls, is_lab = x1.to(CFG.DEVICE), x2.to(CFG.DEVICE), lbls.to(CFG.DEVICE), is_lab.to(CFG.DEVICE)
            with autocast(enabled=CFG.AMP):
                logits, z1 = model(x1)
                _, z2 = model(x2)
                if isinstance(criterion, CombinedLoss) or isinstance(criterion, LossVariantA5):
                    loss, *_ = criterion(logits, lbls, z1, z2, is_lab)
                elif isinstance(criterion, nn.BCEWithLogitsLoss):
                    loss = criterion(logits[:, 1], lbls.float())
                else:
                    loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        _, va_m, *_ = evaluate(model, va_loader, CFG.DEVICE)
        sch.step()
        times.append(time.time() - t0)
        if va_m["f1"] > best_f1:
            best_f1 = va_m["f1"]
            best_state = deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 5:
                break
    model.load_state_dict(best_state)
    _, te_m, _, _, _ = evaluate(model, te_loader, CFG.DEVICE)
    te_m["params_M"] = count_params(model) / 1e6
    te_m["flops_G"] = count_flops(model, CFG.DEVICE)
    te_m["epoch_time_s"] = float(np.mean(times))
    te_m["total_train_min"] = float(np.sum(times) / 60.0)
    te_m["infer_ms"] = measure_inference_latency(model, CFG.DEVICE)
    te_m["throughput_img_s"] = 1000.0 / te_m["infer_ms"] if te_m["infer_ms"] > 0 else float("nan")
    te_m["variant"] = tag
    return te_m


def ablation_A5_learning(tr_loader, va_loader, te_loader, save_dir):
    all_labels = [lbl.item() for _, _, lbl, _ in tr_loader.dataset]
    counts = np.bincount(all_labels)
    cls_w = torch.tensor([1.0, counts[0] / counts[1]], dtype=torch.float32).to(CFG.DEVICE)
    rows = []
    for s in ["supervised", "consistency", "pmcl"]:
        model = SickleLiteMorphFormer()
        crit = LossVariantA5(s, cls_w)
        result = run_custom_loss_experiment(model, tr_loader, va_loader, te_loader, crit, save_dir, f"A5_{s}", CFG.ABLATION_QUICK_EPOCHS)
        result["variant"] = s
        rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "ablation_A5_learning.csv"), index=False)
    save_bar(df, "f1", save_dir, "A5_learning")
    return df


def ablation_A6_loss(tr_loader, va_loader, te_loader, save_dir):
    all_labels = [lbl.item() for _, _, lbl, _ in tr_loader.dataset]
    counts = np.bincount(all_labels)
    cls_w = torch.tensor([1.0, counts[0] / counts[1]], dtype=torch.float32).to(CFG.DEVICE)
    variants = {
        "ce": nn.CrossEntropyLoss(),
        "weighted_ce": nn.CrossEntropyLoss(weight=cls_w),
        "focal": SimpleFocalLoss(gamma=2.0, weight=cls_w),
        "weighted_ce_ls": nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.1),
        "combined": CombinedLoss(cls_w, CFG.LAMBDA_CONS, CFG.LAMBDA_ATTN),
    }
    rows = []
    for name, crit in variants.items():
        model = SickleLiteMorphFormer()
        result = run_custom_loss_experiment(model, tr_loader, va_loader, te_loader, crit, save_dir, f"A6_{name}", CFG.ABLATION_QUICK_EPOCHS)
        result["variant"] = name
        rows.append(result)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "ablation_A6_loss.csv"), index=False)
    save_bar(df, "f1", save_dir, "A6_loss")
    return df


def save_summary(all_results: Dict[str, Dict], save_dir: str, file_name: str = "results_summary.csv"):
    rows = []
    for run_name, metrics in all_results.items():
        row = {"run": run_name}
        for c in METRIC_COLS:
            row[c] = metrics.get(c, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, file_name)
    df.to_csv(path, index=False)
    logger.info(f"Summary saved: {path}")
    return df


def run_train(args, tr_loader, va_loader, te_loader):
    model = SickleLiteMorphFormer().to(CFG.DEVICE)
    metrics, trained_model, _ = train_full(model, tr_loader, va_loader, te_loader, run_name="SickleLite", save_dir=CFG.OUTPUT_DIR)
    save_gradcam_grid(trained_model, te_loader.dataset, CFG.OUTPUT_DIR, "train_gradcam_proposed")
    save_summary({"SickleLite-MorphFormer": metrics}, CFG.OUTPUT_DIR, "train_summary.csv")


def run_benchmark(args, tr_loader, va_loader, te_loader):
    df = benchmark_comparison(tr_loader, va_loader, te_loader, CFG.OUTPUT_DIR)
    benchmark_dict = {r["variant"]: r for _, r in df.iterrows()}
    save_summary(benchmark_dict, CFG.OUTPUT_DIR, "benchmark_summary.csv")


def run_ablation(args, tr_loader, va_loader, te_loader):
    dfs = {
        "A1_backbone": ablation_A1_backbone(tr_loader, va_loader, te_loader, CFG.OUTPUT_DIR),
        "A2_dsrm": ablation_A2_dsrm(tr_loader, va_loader, te_loader, CFG.OUTPUT_DIR),
        "A3_context": ablation_A3_context(tr_loader, va_loader, te_loader, CFG.OUTPUT_DIR),
        "A4_fusion": ablation_A4_fusion(tr_loader, va_loader, te_loader, CFG.OUTPUT_DIR),
        "A5_learning": ablation_A5_learning(tr_loader, va_loader, te_loader, CFG.OUTPUT_DIR),
        "A6_loss": ablation_A6_loss(tr_loader, va_loader, te_loader, CFG.OUTPUT_DIR),
    }
    merged = pd.concat(dfs.values(), ignore_index=True)
    merged.to_csv(os.path.join(CFG.OUTPUT_DIR, "all_ablation_results.csv"), index=False)
    save_bar(merged, "f1", CFG.OUTPUT_DIR, "all_ablation_overview")
    logger.info("All 6 ablations finished.")


def run_all(args, tr_loader, va_loader, te_loader):
    logger.info("Running full package: train + benchmark + ablations")
    run_train(args, tr_loader, va_loader, te_loader)
    run_benchmark(args, tr_loader, va_loader, te_loader)
    run_ablation(args, tr_loader, va_loader, te_loader)


def run_eval(args):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in eval mode")
    paths, labels, tags = collect_images(args.data_root)
    _, _, te_data = stratified_split(paths, labels, tags, seed=CFG.SEED)
    te_set = SickleDataset(*te_data, mode="test")
    te_loader = DataLoader(te_set, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    model = SickleLiteMorphFormer().to(CFG.DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=CFG.DEVICE)
    model.load_state_dict(ckpt["model"])
    _, te_m, y_true, y_pred, y_prob = evaluate(model, te_loader, CFG.DEVICE)
    te_m["infer_ms"] = measure_inference_latency(model, CFG.DEVICE)
    te_m["throughput_img_s"] = 1000.0 / te_m["infer_ms"] if te_m["infer_ms"] > 0 else float("nan")
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in te_m.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k:20s}: {v}")
    save_confusion_matrix(y_true, y_pred, CFG.OUTPUT_DIR, "eval")
    save_roc_pr(y_true, y_prob, CFG.OUTPUT_DIR, "eval")


def parse_args():
    p = argparse.ArgumentParser(description="SickleLite-MorphFormer updated benchmark + ablation pipeline")
    p.add_argument("--data_root", type=str, default=r"D:\sickle\dataset")
    p.add_argument("--mode", type=str, default="train", choices=["train", "benchmark", "ablation", "all", "eval"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CFG.DATA_ROOT = args.data_root
    CFG.OUTPUT_DIR = args.output_dir
    CFG.IMG_SIZE = args.img_size
    CFG.BATCH_SIZE = args.batch_size
    CFG.NUM_EPOCHS = args.epochs
    CFG.LR = args.lr
    CFG.SEED = args.seed
    CFG.NUM_WORKERS = args.workers

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(CFG.OUTPUT_DIR, "figures"), exist_ok=True)
    logger = get_logger("sickle", os.path.join(CFG.OUTPUT_DIR, "logs", "training.log"))
    seed_everything(CFG.SEED)
    logger.info(f"Mode: {args.mode} | Device: {CFG.DEVICE} | Data: {CFG.DATA_ROOT}")

    if args.mode == "eval":
        run_eval(args)
        sys.exit(0)

    paths, labels, tags = collect_images(args.data_root)
    tr_data, va_data, te_data = stratified_split(paths, labels, tags, seed=CFG.SEED)
    tr_loader, va_loader, te_loader = get_loaders(tr_data, va_data, te_data)
    logger.info(f"Train={len(tr_data[0])} | Val={len(va_data[0])} | Test={len(te_data[0])}")

    if args.mode == "train":
        run_train(args, tr_loader, va_loader, te_loader)
    elif args.mode == "benchmark":
        run_benchmark(args, tr_loader, va_loader, te_loader)
    elif args.mode == "ablation":
        run_ablation(args, tr_loader, va_loader, te_loader)
    elif args.mode == "all":
        run_all(args, tr_loader, va_loader, te_loader)
