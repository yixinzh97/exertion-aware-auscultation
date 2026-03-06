"""
Exertion-Aware Dual Bayesian ResNet for 3-Class Murmur Classification
======================================================================
Classes:
    0 - No Murmur (NM)
    1 - Exercise-Induced Murmur (EM)
    2 - Pathological Murmur (PM)

Architecture: BayesianResNet18 with Monte Carlo Dropout
Features:     Mel Spectrogram (64) + RMS (1) + Spectral Centroid/PSD proxy (1) = 66 channels

Usage:
    python dual_bayesian.py \
        --circor_csv /path/to/training_data.csv \
        --circor_wav /path/to/training_data \
        --exercise_csv /path/to/post_exercise_labels.csv \
        --exercise_wav /path/to/eko_PCG \
        --mode train          # or: crossval, evaluate
"""

import os
import argparse
import pickle
import time

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIG
# ============================================================================
SR = 4000
SEGMENT_LEN = 4        # seconds
STRIDE = 1             # seconds
N_MELS = 64
FMIN, FMAX = 10, 2000
N_FFT = int(0.025 * SR)   # 25ms window
HOP = int(0.010 * SR)     # 10ms hop
EPS = 1e-10

LABEL_NAMES = {0: "No Murmur", 1: "Exercise-Induced Murmur", 2: "Pathological Murmur"}

# ============================================================================
# DEVICE
# ============================================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ============================================================================
# FEATURE EXTRACTION: MEL + RMS + PSD (66 channels)
# ============================================================================
def extract_mel_psd_rms(wav_file, sr=SR, n_mels=N_MELS, segment_len=SEGMENT_LEN):
    """
    Extract integrated features from a PCG WAV file.

    Segments the recording into overlapping windows and extracts:
        - 64-channel log-Mel spectrogram
        - 1-channel RMS energy (normalized)
        - 1-channel Spectral Centroid as PSD proxy (normalized)

    Args:
        wav_file (str): Path to .wav file.
        sr (int): Target sample rate. Default 4000 Hz.
        n_mels (int): Number of Mel filter banks. Default 64.
        segment_len (float): Segment length in seconds. Default 4s.

    Returns:
        np.ndarray of shape (n_segments, 66, T), or None on error.
    """
    try:
        y, _ = librosa.load(wav_file, sr=sr, mono=True)
        segment_samples = int(sr * segment_len)
        integrated_specs = []

        for i in range(0, len(y) - segment_samples, segment_samples // 2):
            seg = y[i:i + segment_samples]

            # 1. Mel Spectrogram (64, T)
            mel = librosa.feature.melspectrogram(
                y=seg, sr=sr, n_mels=n_mels, n_fft=N_FFT,
                hop_length=HOP, fmin=FMIN, fmax=FMAX
            )
            mel_db = librosa.power_to_db(mel + EPS, ref=np.max)

            # 2. RMS Energy (1, T)
            rms = librosa.feature.rms(y=seg, hop_length=HOP)[0]
            rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + EPS)

            # 3. Spectral Centroid as PSD proxy (1, T)
            sc = librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=HOP)[0]
            sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + EPS)

            # Align all features to Mel time dimension T
            T = mel_db.shape[1]
            rms_r = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(rms_norm)), rms_norm)
            sc_r = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(sc_norm)), sc_norm)

            # Stack: (66, T)
            integrated = np.vstack([
                mel_db,                      # (64, T)
                rms_r[np.newaxis, :],         # (1, T)
                sc_r[np.newaxis, :]           # (1, T)
            ]).astype(np.float32)

            integrated_specs.append(integrated)

        return np.stack(integrated_specs) if integrated_specs else None

    except Exception as e:
        print(f"[WARN] Error loading {wav_file}: {e}")
        return None

# ============================================================================
# DATASET
# ============================================================================
class MurmurDataset(Dataset):
    """
    PyTorch Dataset for 3-class murmur classification.

    Args:
        data_df (pd.DataFrame): Must contain 'file' and 'label' columns.
            label: 0=NM, 1=EM, 2=PM
    """
    def __init__(self, data_df):
        self.data = data_df.reset_index(drop=True)
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        wav_file = row["file"]

        if wav_file not in self.cache:
            specs = extract_mel_psd_rms(wav_file)
            self.cache[wav_file] = specs

        specs = self.cache[wav_file]
        if specs is None or len(specs) == 0:
            specs = np.zeros((1, 66, 239), dtype=np.float32)

        return {
            "specs": torch.FloatTensor(specs),   # (n_segments, 66, T)
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "file": wav_file,
        }


def collate_fn(batch):
    """Pad variable-length segment sequences within a batch."""
    specs_list = [item["specs"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    files = [item["file"] for item in batch]

    max_segs = max(s.shape[0] for s in specs_list)
    padded = []
    for s in specs_list:
        if s.shape[0] < max_segs:
            pad = torch.zeros(max_segs - s.shape[0], s.shape[1], s.shape[2])
            s = torch.cat([s, pad], dim=0)
        padded.append(s)

    return torch.stack(padded), labels, files   # (B, max_segs, 66, T), (B,), list

# ============================================================================
# MODEL: BAYESIAN RESNET18 (66-channel input)
# ============================================================================
class BayesianResNet18(nn.Module):
    """
    Bayesian ResNet18 for 3-class murmur classification with MC Dropout.

    Accepts 66-channel feature maps (Mel + RMS + PSD).
    The first conv layer is re-initialized to accept 66 input channels,
    with pretrained weights averaged across channels for transfer learning.

    Args:
        num_classes (int): Number of output classes. Default 3.
        dropout_p (float): MC Dropout probability. Default 0.3.
        pretrained (bool): Use ImageNet pretrained weights. Default True.
    """
    def __init__(self, num_classes=3, dropout_p=0.3, pretrained=True):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Replace first conv to accept 66 channels
        orig_conv = base.conv1
        base.conv1 = nn.Conv2d(
            in_channels=66,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias,
        )
        # Initialize: average pretrained 3-channel weights → repeat for 66 channels
        with torch.no_grad():
            avg_w = orig_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            base.conv1.weight = nn.Parameter(avg_w.repeat(1, 66, 1, 1) / 66)

        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(512, num_classes)
        self.enable_dropout = False   # flag for MC inference

    def forward(self, x):
        """
        Args:
            x: (B, 66, H, W)
        Returns:
            logits: (B, num_classes)
        """
        feat = self.feature_extractor(x)   # (B, 512, 1, 1)
        feat = torch.flatten(feat, 1)       # (B, 512)
        if self.training or self.enable_dropout:
            feat = self.dropout(feat)
        return self.classifier(feat)

    def predict_mc(self, x, mc_passes=10):
        """
        Monte Carlo Dropout inference.

        Args:
            x: (B, 66, H, W)
            mc_passes (int): Number of stochastic forward passes.

        Returns:
            mean (Tensor): Mean softmax probabilities (B, num_classes)
            var (Tensor): Variance across passes (B, num_classes)
        """
        self.eval()
        self.enable_dropout = True
        preds = []
        with torch.no_grad():
            for _ in range(mc_passes):
                probs = F.softmax(self.forward(x), dim=1)
                preds.append(probs.unsqueeze(0))
        preds = torch.cat(preds, dim=0)   # (mc, B, C)
        self.enable_dropout = False
        return preds.mean(dim=0), preds.var(dim=0)

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================
def _forward_batch(model, specs, device):
    """Average segment-level logits to recording level."""
    specs = specs.to(device)
    B, n_segs, C, T = specs.shape
    specs_flat = specs.view(B * n_segs, C, T).unsqueeze(-1)   # (B*S, 66, T, 1)
    logits = model(specs_flat)                                  # (B*S, 3)
    return logits.view(B, n_segs, -1).mean(dim=1)              # (B, 3)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for specs, labels, _ in loader:
        labels = labels.to(device)
        optimizer.zero_grad()
        logits_avg = _forward_batch(model, specs, device)
        loss = criterion(logits_avg, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits_avg.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for specs, labels, _ in loader:
        logits_avg = _forward_batch(model, specs, device)
        all_preds.extend(logits_avg.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, device,
                epochs=20, lr=1e-4, save_path="model_dual_bayesian.pth",
                patience=10):
    """Full training loop with early stopping and best-model checkpointing."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_val_acc, patience_counter = 0, 0

    print(f"\n{'='*60}")
    print(f"Training Exertion-Aware Dual Bayesian ResNet18")
    print(f"  Epochs={epochs}  LR={lr}  Device={device}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_acc = (val_preds == val_labels).mean()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            marker = " ✓ BEST"
        else:
            patience_counter += 1

        print(f"Ep {epoch+1:3d}/{epochs} | Loss {tr_loss:.4f} | "
              f"TrainAcc {tr_acc:.3f} | ValAcc {val_acc:.3f}{marker}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")
    return best_val_acc

# ============================================================================
# DATA LOADING
# ============================================================================
def build_manifest(circor_csv, circor_wav, exercise_csv, exercise_wav):
    """
    Build a unified manifest DataFrame from CirCor DB and Post-Exercise DB.

    Label mapping:
        CirCor  Absent  → 0 (NM)
        CirCor  Present → 2 (PM)
        Exercise No Murmur Detected → 0 (NM)
        Exercise Murmur Detected    → 1 (EM)

    Returns:
        pd.DataFrame with columns: file, label, source, group
    """
    manifest = []

    # --- CirCor ---
    df_c = pd.read_csv(circor_csv)
    circor_wav = str(circor_wav)
    for _, row in df_c.iterrows():
        pid = str(row["Patient ID"]).strip()
        murmur = str(row["Murmur"]).strip()
        if murmur == "Absent":
            label = 0
        elif murmur == "Present":
            label = 2
        else:
            continue
        # Try AV location as primary; extend here for other locations if needed
        wav = os.path.join(circor_wav, f"{pid}_AV.wav")
        if os.path.exists(wav):
            manifest.append({"file": wav, "label": label, "source": "circor", "group": f"c_{pid}"})

    # --- Post-Exercise ---
    df_e = pd.read_csv(exercise_csv).dropna(subset=["original_filename"])
    exercise_wav = str(exercise_wav)
    for _, row in df_e.iterrows():
        murmur = str(row["Murmur_Groundtruth"]).strip()
        if "No Murmur" in murmur:
            label = 0
        elif "Murmur Detected" in murmur:
            label = 1
        else:
            continue
        fname = str(row["original_filename"]).strip()
        wav = os.path.join(exercise_wav, fname)
        if os.path.exists(wav):
            subj = os.path.splitext(fname)[0]
            manifest.append({"file": wav, "label": label, "source": "exercise", "group": f"ex_{subj}"})

    df = pd.DataFrame(manifest)
    print(f"Manifest: {len(df)} samples | "
          f"NM={( df['label']==0).sum()} | "
          f"EM={(df['label']==1).sum()} | "
          f"PM={(df['label']==2).sum()}")
    return df


def get_splits(df, split_pkl=None, random_state=42):
    """
    Return train/val/test DataFrames.

    If split_pkl exists, applies saved proportions (for reproducibility).
    Otherwise creates a stratified group split (80/10/10).
    """
    if split_pkl and os.path.exists(split_pkl):
        with open(split_pkl, "rb") as f:
            saved = pickle.load(f)
        n = len(saved["train_idx"]) + len(saved["val_idx"]) + len(saved["test_idx"])
        train_r = len(saved["train_idx"]) / n
        val_r = len(saved["val_idx"]) / n
        test_r = len(saved["test_idx"]) / n
        print(f"Loaded split ratios from {split_pkl}: "
              f"train={train_r:.2f} val={val_r:.2f} test={test_r:.2f}")
    else:
        train_r, val_r, test_r = 0.80, 0.10, 0.10
        print("No split file found — using default 80/10/10 stratified split.")

    idx = np.arange(len(df))
    labels = df["label"].values
    tr_idx, tmp_idx = train_test_split(idx, test_size=1 - train_r,
                                        random_state=random_state, stratify=labels)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=test_r / (val_r + test_r),
                                        random_state=random_state, stratify=labels[tmp_idx])
    return (df.iloc[tr_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[te_idx].reset_index(drop=True))


def make_loaders(train_df, val_df, test_df, batch_size=8):
    train_ds = MurmurDataset(train_df)
    val_ds = MurmurDataset(val_df)
    test_ds = MurmurDataset(test_df)
    kwargs = dict(collate_fn=collate_fn, num_workers=0)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kwargs),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs))

# ============================================================================
# CROSS-VALIDATION
# ============================================================================
def run_crossval(df, device, n_folds=5, epochs=30, lr=1e-4, random_state=42):
    """
    5-fold stratified cross-validation on train+val split.

    Returns:
        pd.DataFrame with per-fold accuracy and per-class recall/F1.
    """
    # Hold out a fixed test set first
    train_val_df, test_df = train_test_split(df, test_size=0.10,
                                              random_state=random_state,
                                              stratify=df["label"])
    train_val_df = train_val_df.reset_index(drop=True)
    y = train_val_df["label"].values

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_results = []

    print(f"\n{'='*60}")
    print(f"5-FOLD CROSS-VALIDATION  ({len(train_val_df)} samples in CV pool)")
    print(f"{'='*60}")

    for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(np.arange(len(train_val_df)), y)):
        print(f"\n--- Fold {fold_idx+1}/{n_folds} ---")
        fold_tr = train_val_df.iloc[tr_idx].reset_index(drop=True)
        fold_val = train_val_df.iloc[val_idx].reset_index(drop=True)

        tr_loader, val_loader, _ = make_loaders(fold_tr, fold_val, fold_val)

        model = BayesianResNet18(num_classes=3, dropout_p=0.3, pretrained=True).to(device)
        train_model(model, tr_loader, val_loader, device,
                    epochs=epochs, lr=lr,
                    save_path=f"model_fold{fold_idx+1}.pth",
                    patience=10)

        preds, labels = evaluate(model, val_loader, device)
        acc = (preds == labels).mean()
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2])

        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": acc,
            "recall_NM": rec[0], "recall_EM": rec[1], "recall_PM": rec[2],
            "f1_NM": f1[0], "f1_EM": f1[1], "f1_PM": f1[2],
        })
        print(f"Fold {fold_idx+1}: Acc={acc:.4f} | "
              f"Recall NM={rec[0]:.4f} EM={rec[1]:.4f} PM={rec[2]:.4f}")

    results_df = pd.DataFrame(fold_results)
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for col in ["accuracy", "recall_NM", "recall_EM", "recall_PM"]:
        print(f"  {col:20s}: {results_df[col].mean():.4f} ± {results_df[col].std():.4f}")
    results_df.to_csv("5fold_cv_results.csv", index=False)
    print("\nSaved: 5fold_cv_results.csv")
    return results_df, test_df

# ============================================================================
# EVALUATION REPORT
# ============================================================================
def evaluate_and_report(model, loader, device, save_fig="confusion_matrix.png"):
    """Print full classification report and save confusion matrix figure."""
    preds, labels = evaluate(model, loader, device)
    acc = (preds == labels).mean()

    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(labels, preds,
                                  target_names=["No Murmur", "Exercise-Induced", "Pathological"],
                                  zero_division=0))

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    label_names = ["No Murmur", "Exercise-Induced", "Pathological"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Exertion-Aware Dual Bayesian ResNet18")
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    print(f"Saved: {save_fig}")
    return acc, cm

# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Exertion-Aware Dual Bayesian ResNet")
    p.add_argument("--circor_csv",    required=True, help="CirCor training_data.csv")
    p.add_argument("--circor_wav",    required=True, help="CirCor WAV directory")
    p.add_argument("--exercise_csv",  required=True, help="Post-exercise labels CSV")
    p.add_argument("--exercise_wav",  required=True, help="Post-exercise WAV directory")
    p.add_argument("--split_pkl",     default=None,  help="Saved split indices .pkl (optional)")
    p.add_argument("--mode",          default="train",
                   choices=["train", "crossval", "evaluate"],
                   help="Run mode (default: train)")
    p.add_argument("--checkpoint",    default="model_dual_bayesian.pth",
                   help="Path to save/load model checkpoint")
    p.add_argument("--epochs",        type=int, default=20)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--batch_size",    type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    # Build dataset
    df = build_manifest(args.circor_csv, args.circor_wav,
                        args.exercise_csv, args.exercise_wav)

    if args.mode == "crossval":
        run_crossval(df, device, n_folds=5, epochs=args.epochs, lr=args.lr)

    elif args.mode == "train":
        train_df, val_df, test_df = get_splits(df, split_pkl=args.split_pkl)
        tr_loader, val_loader, te_loader = make_loaders(train_df, val_df, test_df,
                                                         batch_size=args.batch_size)
        model = BayesianResNet18(num_classes=3, dropout_p=0.3, pretrained=True).to(device)
        train_model(model, tr_loader, val_loader, device,
                    epochs=args.epochs, lr=args.lr, save_path=args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        evaluate_and_report(model, te_loader, device)

    elif args.mode == "evaluate":
        _, _, test_df = get_splits(df, split_pkl=args.split_pkl)
        _, _, te_loader = make_loaders(test_df, test_df, test_df, batch_size=args.batch_size)
        model = BayesianResNet18(num_classes=3, dropout_p=0.3, pretrained=True).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        evaluate_and_report(model, te_loader, device)


if __name__ == "__main__":
    main()
