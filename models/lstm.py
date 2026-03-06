"""
HM-Detect: Heart Murmur Detection with Stacked LSTM
=====================================================
Replicates the HM-Detect baseline from Sivaraman & Xiao (IEEE SiPS 2024).

Features:  Spectral Sub-band Energies (SSE) + Spectral Sub-band Centroids (SSC)
           18 sub-bands × 2 = 36 features per frame
Model:     Bidirectional 3-layer stacked LSTM with Focal Loss
Task:      Binary classification — 0: No Murmur, 1: Murmur Present

Usage:
    python lstm.py \
        --circor_csv  /path/to/training_data.csv \
        --circor_wav  /path/to/training_data \
        --mode train          # or: evaluate
"""

import os
import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.signal import windows
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
# FEATURE EXTRACTION: SSE + SSC (36-dim per frame)
# ============================================================================
class SpectralFeatureExtractor:
    """
    Extract Spectral Sub-band Energies (SSE) and Spectral Sub-band Centroids (SSC).

    Each audio frame produces a 36-dim vector: 18 SSE + 18 SSC.

    Args:
        sr (int): Sample rate. Default 4000 Hz.
        n_subbands (int): Number of spectral sub-bands. Default 18.
        window_ms (int): Analysis window length in ms. Default 25.
        hop_ms (int): Hop length in ms. Default 10.
    """
    def __init__(self, sr=4000, n_subbands=18, window_ms=25, hop_ms=10):
        self.sr = sr
        self.n_subbands = n_subbands
        self.window_ms = window_ms
        self.hop_ms = hop_ms
        self.window_samples = int(sr * window_ms / 1000)
        self.hop_samples = int(sr * hop_ms / 1000)
        self.n_fft = 2 ** int(np.ceil(np.log2(self.window_samples)))
        self._create_filterbank()

    def _create_filterbank(self):
        """Build mel-spaced triangular filter bank."""
        nyquist = self.sr / 2
        freq_bins = np.linspace(0, nyquist, self.n_fft // 2 + 1)
        mel_edges = np.linspace(0, 2595 * np.log10(1 + nyquist / 700),
                                self.n_subbands + 2)
        hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)

        self.filterbank = []
        for i in range(self.n_subbands):
            left, center, right = hz_edges[i], hz_edges[i+1], hz_edges[i+2]
            filt = np.zeros(len(freq_bins))
            if i == 0:
                filt[freq_bins <= center] = 1.0
                mask = (freq_bins > center) & (freq_bins <= right)
                filt[mask] = (right - freq_bins[mask]) / (right - center + 1e-10)
            elif i == self.n_subbands - 1:
                mask = (freq_bins >= left) & (freq_bins < center)
                filt[mask] = (freq_bins[mask] - left) / (center - left + 1e-10)
                filt[freq_bins >= center] = 1.0
            else:
                mask_l = (freq_bins >= left) & (freq_bins < center)
                filt[mask_l] = (freq_bins[mask_l] - left) / (center - left + 1e-10)
                mask_r = (freq_bins > center) & (freq_bins <= right)
                filt[mask_r] = (right - freq_bins[mask_r]) / (right - center + 1e-10)
                filt[freq_bins == center] = 1.0
            self.filterbank.append(filt)
        self.filterbank = np.array(self.filterbank)

    def extract(self, audio):
        """
        Extract SSE + SSC features.

        Args:
            audio (np.ndarray): 1-D float32 audio array.

        Returns:
            np.ndarray: Shape (n_frames, 36). Returns empty (0, 36) on error.
        """
        try:
            win = windows.hamming(self.window_samples)
            freq_bins = np.fft.rfftfreq(self.n_fft, 1 / self.sr)
            n_frames = 1 + (len(audio) - self.window_samples) // self.hop_samples
            sse_list, ssc_list = [], []

            for i in range(n_frames):
                start = i * self.hop_samples
                frame = audio[start:start + self.window_samples]
                if len(frame) < self.window_samples:
                    break
                spec = np.abs(np.fft.rfft(frame * win, self.n_fft)) ** 2

                sse_row, ssc_row = [], []
                for filt in self.filterbank:
                    weighted = spec[:len(filt)] * filt
                    sse_row.append(np.sum(weighted))
                    denom = np.sum(weighted) + 1e-10
                    ssc_row.append(np.sum(weighted * freq_bins[:len(filt)]) / denom)
                sse_list.append(sse_row)
                ssc_list.append(ssc_row)

            if not sse_list:
                return np.zeros((0, 36), dtype=np.float32)

            sse = np.log10(np.array(sse_list) + 1e-10)
            ssc = np.clip(2 * (np.array(ssc_list) / (self.sr / 2)) - 1, -1, 1)
            return np.hstack([sse, ssc]).astype(np.float32)

        except Exception as e:
            print(f"[WARN] Feature extraction error: {e}")
            return np.zeros((0, 36), dtype=np.float32)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class HMDetectLSTM(nn.Module):
    """
    Bidirectional 3-layer stacked LSTM for binary murmur detection.

    Architecture (Model B from HM-Detect):
        BiLSTM(256) → BiLSTM(128) → LSTM(64) → FC(128) → FC(64) → FC(2)

    Args:
        input_size (int): Input feature dimension. Default 36.
        dropout (float): Dropout probability. Default 0.5.
        num_classes (int): Output classes. Default 2 (binary).
    """
    def __init__(self, input_size=36, dropout=0.5, num_classes=2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 128, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(256, 64, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, T, 36)
        Returns:
            logits: (B, num_classes)
        """
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x[:, -1, :])   # last timestep
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc_out(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced binary classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha (Tensor): Per-class weights. Shape (num_classes,).
        gamma (float): Focusing parameter. Default 2.5.
    """
    def __init__(self, alpha=None, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()

# ============================================================================
# DATASET
# ============================================================================
def pad_or_truncate(features_list, max_frames=400):
    """Pad/truncate all sequences to max_frames."""
    out = []
    for f in features_list:
        if f.shape[0] < max_frames:
            pad = np.zeros((max_frames - f.shape[0], f.shape[1]), dtype=np.float32)
            f = np.vstack([f, pad])
        else:
            f = f[:max_frames]
        out.append(f)
    return out


class PCGDataset(Dataset):
    """
    PCG dataset with optional on-the-fly augmentation.

    Args:
        features (list): List of (T, 36) float32 arrays.
        labels (list): Integer labels (0 or 1).
        augment (bool): Apply speed/noise augmentation. Default False.
        aug_prob (float): Augmentation probability per sample. Default 0.5.
    """
    def __init__(self, features, labels, augment=False, aug_prob=0.5):
        self.features = features
        self.labels = labels
        self.augment = augment
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.features)

    def _augment(self, f):
        if np.random.rand() > self.aug_prob:
            return f
        if np.random.rand() < 0.5:
            # Speed perturbation
            rate = 1.0 + np.random.uniform(-0.15, 0.15)
            T = f.shape[0]
            src_idx = np.linspace(0, T - 1, int(T * rate))
            tgt_idx = np.linspace(0, T - 1, T)
            f = np.stack([np.interp(tgt_idx, src_idx, np.interp(src_idx,
                          np.arange(T), f[:, c])) for c in range(f.shape[1])], axis=1)
        else:
            # Gaussian noise
            f = f + np.random.normal(0, 0.15 * np.std(f), f.shape).astype(np.float32)
        return f.astype(np.float32)

    def __getitem__(self, idx):
        f = self.features[idx].copy()
        if self.augment:
            f = self._augment(f)
        return torch.FloatTensor(f), torch.tensor(self.labels[idx], dtype=torch.long)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_circor(csv_path, audio_dir, extractor, sr=4000):
    """
    Load CirCor DigiScope dataset and extract SSE+SSC features.

    Label mapping:
        Absent  → 0 (No Murmur)
        Present → 1 (Murmur)

    Args:
        csv_path (str): Path to training_data.csv.
        audio_dir (str): Directory containing .wav files.
        extractor (SpectralFeatureExtractor): Feature extractor instance.
        sr (int): Target sample rate. Default 4000.

    Returns:
        features_list (list): List of (T, 36) arrays.
        labels_list (list): Binary integer labels.
    """
    df = pd.read_csv(csv_path)
    features_list, labels_list = [], []

    for _, row in df.iterrows():
        pid = str(row["Patient ID"]).strip()
        murmur = str(row["Murmur"]).strip().lower()
        if murmur == "absent":
            label = 0
        elif murmur == "present":
            label = 1
        else:
            continue

        for loc in str(row["Recording locations:"]).strip().split("+"):
            loc = loc.strip()
            if not loc:
                continue
            wav_path = Path(audio_dir) / f"{pid}_{loc}.wav"
            if not wav_path.exists():
                continue
            try:
                sr_file, audio = wavfile.read(str(wav_path))
                if sr_file != sr:
                    audio = scipy_signal.resample(audio, int(len(audio) * sr / sr_file))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
                audio /= (np.max(np.abs(audio)) + 1e-10)
                if len(audio) < sr:
                    continue
                feats = extractor.extract(audio)
                if feats.shape[0] == 0:
                    continue
                features_list.append(feats)
                labels_list.append(label)
            except Exception:
                continue

    print(f"Loaded {len(features_list)} recordings | "
          f"No Murmur: {labels_list.count(0)} | Murmur: {labels_list.count(1)}")
    return features_list, labels_list


def balance_and_split(features, labels, no_murmur_fraction=0.33,
                      test_size=0.25, random_state=42):
    """
    Under-sample 'No Murmur' class and create stratified train/test split.

    Args:
        no_murmur_fraction (float): Fraction of NM samples to keep. Default 0.33.
        test_size (float): Fraction for test set. Default 0.25.

    Returns:
        X_train, X_test, y_train, y_test
    """
    nm_idx = [i for i, l in enumerate(labels) if l == 0]
    m_idx  = [i for i, l in enumerate(labels) if l == 1]
    keep_nm = np.random.choice(nm_idx, int(len(nm_idx) * no_murmur_fraction), replace=False)
    balanced_idx = list(keep_nm) + m_idx
    bal_X = [features[i] for i in balanced_idx]
    bal_y = [labels[i] for i in balanced_idx]
    print(f"Balanced: {len(bal_y)} samples | NM: {bal_y.count(0)} | M: {bal_y.count(1)}")
    return train_test_split(bal_X, bal_y, test_size=test_size,
                            random_state=random_state, stratify=bal_y)

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        total_loss += criterion(out, y).item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return total_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, device,
                epochs=50, lr=1e-3, save_path="model_lstm.pth",
                class_weights=None, patience=8):
    """
    Training loop with Focal Loss, LR scheduling, and early stopping.

    Args:
        class_weights (np.ndarray): Per-class weights for Focal Loss.
        patience (int): Early stopping patience. Default 8.
    """
    alpha = None
    if class_weights is not None:
        alpha = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = FocalLoss(alpha=alpha, gamma=2.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                      factor=0.5, patience=4)
    best_val_acc, patience_counter = 0, 0

    print(f"\n{'='*60}")
    print(f"Training HM-Detect LSTM  (Focal Loss + BiLSTM + Augmentation)")
    print(f"  Epochs={epochs}  LR={lr}  Device={device}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(va_acc)

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            marker = " ✓ BEST"
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or marker:
            print(f"Ep {epoch+1:3d}/{epochs} | Loss {tr_loss:.4f} | "
                  f"TrainAcc {tr_acc:.3f} | ValAcc {va_acc:.3f}{marker}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")
    return best_val_acc

# ============================================================================
# EVALUATION REPORT
# ============================================================================
def evaluate_and_report(model, loader, device, criterion=None,
                        save_fig="confusion_matrix_lstm.png"):
    """Print classification report and save confusion matrix figure."""
    if criterion is None:
        criterion = FocalLoss()
    _, acc, preds, labels = eval_epoch(model, loader, criterion, device)

    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(labels, preds,
                                  target_names=["No Murmur", "Murmur Present"],
                                  zero_division=0))
    try:
        roc = roc_auc_score(labels, preds)
        print(f"ROC-AUC: {roc:.4f}")
    except Exception:
        pass

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Murmur", "Murmur Present"],
                yticklabels=["No Murmur", "Murmur Present"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — HM-Detect LSTM")
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    print(f"Saved: {save_fig}")
    return acc, cm

# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="HM-Detect LSTM Baseline")
    p.add_argument("--circor_csv",  required=True, help="CirCor training_data.csv")
    p.add_argument("--circor_wav",  required=True, help="CirCor WAV directory")
    p.add_argument("--mode",        default="train", choices=["train", "evaluate"])
    p.add_argument("--checkpoint",  default="model_lstm.pth")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--max_frames",  type=int,   default=400,
                   help="Truncate/pad sequences to this many frames")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Feature extraction
    extractor = SpectralFeatureExtractor(sr=4000, n_subbands=18,
                                          window_ms=25, hop_ms=10)

    # Load data
    features, labels = load_circor(args.circor_csv, args.circor_wav, extractor)

    # Split
    X_train, X_test, y_train, y_test = balance_and_split(features, labels)

    # Pad / truncate
    X_train_p = pad_or_truncate(X_train, args.max_frames)
    X_test_p  = pad_or_truncate(X_test,  args.max_frames)

    # Class weights for Focal Loss
    counts = np.bincount(y_train)
    cw = len(y_train) / (len(counts) * counts)
    cw = cw / cw.sum() * len(counts)

    # DataLoaders
    train_ds = PCGDataset(X_train_p, y_train, augment=True, aug_prob=0.6)
    test_ds  = PCGDataset(X_test_p,  y_test,  augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = HMDetectLSTM(input_size=36, dropout=0.5, num_classes=2).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.mode == "train":
        train_model(model, train_loader, test_loader, device,
                    epochs=args.epochs, lr=args.lr,
                    save_path=args.checkpoint,
                    class_weights=cw, patience=8)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        evaluate_and_report(model, test_loader, device)

    elif args.mode == "evaluate":
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        evaluate_and_report(model, test_loader, device)


if __name__ == "__main__":
    main()
