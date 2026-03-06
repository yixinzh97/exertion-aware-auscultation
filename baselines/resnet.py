"""
MurmurGrade-Net: ResNet Ensemble for Binary Murmur Classification
==================================================================
Replicates the ResNet murmur grading baseline from Liu et al.
(ICLR Time Series for Health Workshop, 2023).

Architecture: Ensemble of N ResNet models with SE-attention
              + Depthwise Separable Convolutions
Features:     Mel-spectrogram (32 × 239), 3s segments @ 4kHz
Task:         Binary classification — 0: Absent, 1: Present

Data source:  CirCor DigiScope Dataset (reads .txt + .wav files
              directly from the CirCor training_data directory)

Usage:
    python resnet.py \
        --data_dir /path/to/circor/training_data \
        --mode train          # or: evaluate, predict
    
    # Single-file inference after training:
    python resnet.py \
        --data_dir /path/to/circor/training_data \
        --mode predict \
        --audio_path /path/to/recording.wav \
        --checkpoint_dir ./checkpoints
"""

import os
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

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
# FEATURE EXTRACTION: MEL SPECTROGRAM (32 × 239)
# ============================================================================
class PCGPreprocessor:
    """
    Extract log-Mel spectrograms from PCG recordings.

    Each recording is split into non-overlapping 3-second segments.
    Each segment produces a (32, 239) Mel spectrogram.

    Args:
        sr (int): Target sample rate. Default 4000 Hz.
        segment_duration (float): Segment length in seconds. Default 3.0.
        n_mels (int): Number of Mel filter banks. Default 32.
        n_fft (int): FFT size. Default 512.
        fmax (int): Maximum frequency for Mel bank. Default 800 Hz.
    """
    def __init__(self, sr=4000, segment_duration=3.0,
                 n_mels=32, n_fft=512, fmax=800):
        self.sr = sr
        self.segment_duration = segment_duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = int(0.01 * sr)     # 10 ms
        self.win_length = int(0.025 * sr)    # 25 ms
        self.fmax = fmax
        self.samples_per_segment = int(segment_duration * sr)

    def extract(self, audio_path):
        """
        Load a WAV file and extract per-segment Mel spectrograms.

        Args:
            audio_path (str): Path to .wav file.

        Returns:
            np.ndarray of shape (n_segments, 32, 239), or None on error.
        """
        try:
            y, _ = librosa.load(audio_path, sr=self.sr)
            n_segments = len(y) // self.samples_per_segment
            if n_segments == 0:
                return None
            spectrograms = []
            for i in range(n_segments):
                seg = y[i * self.samples_per_segment:(i + 1) * self.samples_per_segment]
                mel = librosa.feature.melspectrogram(
                    y=seg, sr=self.sr, n_mels=self.n_mels,
                    n_fft=self.n_fft, hop_length=self.hop_length,
                    win_length=self.win_length, window="hamming", fmax=self.fmax
                )
                spectrograms.append(librosa.power_to_db(mel, ref=np.max))
            return np.array(spectrograms, dtype=np.float32)
        except Exception:
            return None

# ============================================================================
# DATASET
# ============================================================================
class CircorPCGDataset(Dataset):
    """
    CirCor DigiScope dataset.

    Reads patient labels from .txt files and loads all matching .wav files.
    Label mapping: Absent → 0, Present → 1.

    Args:
        data_dir (str): Directory containing CirCor .txt and .wav files.
        preprocessor (PCGPreprocessor): Feature extractor instance.
    """
    def __init__(self, data_dir, preprocessor=None):
        self.data_dir = data_dir
        self.preprocessor = preprocessor or PCGPreprocessor()
        self.data = []
        self._load_manifest()

    def _parse_label(self, txt_file):
        try:
            with open(txt_file) as f:
                for line in f:
                    if line.startswith("#Murmur:"):
                        status = line.split(":")[1].strip()
                        return 0 if "Absent" in status else 1
        except Exception:
            pass
        return None

    def _load_manifest(self):
        for txt_file in glob.glob(os.path.join(self.data_dir, "*.txt")):
            pid = os.path.basename(txt_file).replace(".txt", "")
            label = self._parse_label(txt_file)
            if label is None:
                continue
            wavs = glob.glob(os.path.join(self.data_dir, f"{pid}_*.wav"))
            if wavs:
                self.data.append({"patient_id": pid,
                                   "audio_files": wavs,
                                   "label": label})
        print(f"Loaded {len(self.data)} patients | "
              f"Absent: {sum(d['label']==0 for d in self.data)} | "
              f"Present: {sum(d['label']==1 for d in self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient = self.data[idx]
        all_specs = []
        for wav in patient["audio_files"]:
            specs = self.preprocessor.extract(wav)
            if specs is not None:
                all_specs.extend(specs)
        if not all_specs:
            return None, None, None
        return (torch.FloatTensor(np.array(all_specs)),
                torch.LongTensor([patient["label"]]),
                patient["patient_id"])


def collate_fn(batch):
    """Skip None samples; return variable-length spec lists."""
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None, None
    specs  = [b[0] for b in batch]
    labels = torch.cat([b[1] for b in batch])
    pids   = [b[2] for b in batch]
    return specs, labels, pids

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class SqueezeExcitation(nn.Module):
    """Channel-wise SE attention block."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s)))).view(b, c, 1, 1)
        return x * s


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class ResidualBlock(nn.Module):
    """Residual block with SE attention and separable convolutions."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = SeparableConv2d(in_ch, out_ch, stride=stride)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = SeparableConv2d(out_ch, out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SqueezeExcitation(out_ch)
        self.skip  = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.skip(x))


class MurmurResNet(nn.Module):
    """
    ResNet with SE-attention and depthwise separable convolutions
    for binary murmur classification.

    Input:  (B, 1, 32, 239)  — single-channel Mel spectrogram segment
    Output: (B, 2)            — logits for [Absent, Present]
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem    = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1  = self._make_layer(32,  64,  2, stride=1)
        self.layer2  = self._make_layer(64,  128, 2, stride=2)
        self.layer3  = self._make_layer(128, 256, 2, stride=2)
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.head    = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# ============================================================================
# ENSEMBLE
# ============================================================================
class MurmurEnsemble:
    """
    Ensemble of N independently trained MurmurResNet models.

    Each model is trained separately with a different random initialisation.
    At inference, predictions are averaged across all models and all segments,
    producing a single per-patient probability vector.

    Args:
        num_models (int): Number of ensemble members. Default 15.
        device (torch.device): Compute device.
    """
    def __init__(self, num_models=15, device=None):
        self.device = device or get_device()
        self.models = [MurmurResNet().to(self.device) for _ in range(num_models)]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _patient_logits(self, model, specs_tensor):
        """Average segment-level softmax probs for one patient."""
        inp = specs_tensor.unsqueeze(1).to(self.device)   # (S, 1, H, W)
        with torch.no_grad():
            return F.softmax(model(inp), dim=1).mean(dim=0, keepdim=True)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train(self, train_loader, val_loader,
              epochs=18, lr=1e-3, checkpoint_dir="./checkpoints"):
        """Train each ensemble member independently."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        criterion = nn.CrossEntropyLoss()

        for m_idx, model in enumerate(self.models):
            print(f"\n--- Model {m_idx+1}/{len(self.models)} ---")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            best_val_acc = 0.0

            for epoch in range(epochs):
                # --- train ---
                model.train()
                tr_loss, tr_correct, tr_total = 0, 0, 0
                for batch in train_loader:
                    if batch[0] is None:
                        continue
                    specs_list, labels, _ = batch
                    outs, lbls = [], []
                    for i, specs in enumerate(specs_list):
                        outs.append(self._patient_logits(model, specs))
                        lbls.append(labels[i:i+1])
                    if not outs:
                        continue
                    batch_out = torch.cat(outs).to(self.device)
                    batch_lbl = torch.cat(lbls).to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(batch_out, batch_lbl)
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item()
                    tr_correct += (batch_out.argmax(1) == batch_lbl).sum().item()
                    tr_total += len(batch_lbl)

                # --- validate every 5 epochs ---
                if (epoch + 1) % 5 == 0:
                    model.eval()
                    val_correct, val_total = 0, 0
                    for batch in val_loader:
                        if batch[0] is None:
                            continue
                        specs_list, labels, _ = batch
                        outs, lbls = [], []
                        for i, specs in enumerate(specs_list):
                            outs.append(self._patient_logits(model, specs))
                            lbls.append(labels[i:i+1])
                        if not outs:
                            continue
                        batch_out = torch.cat(outs).to(self.device)
                        batch_lbl = torch.cat(lbls).to(self.device)
                        val_correct += (batch_out.argmax(1) == batch_lbl).sum().item()
                        val_total += len(batch_lbl)

                    val_acc = val_correct / val_total if val_total else 0
                    tr_acc  = tr_correct  / tr_total  if tr_total  else 0
                    print(f"  Ep {epoch+1:3d}/{epochs} | "
                          f"TrainAcc {tr_acc:.3f} | ValAcc {val_acc:.3f}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        ckpt = os.path.join(checkpoint_dir,
                                            f"model_{m_idx+1}.pth")
                        torch.save(model.state_dict(), ckpt)

            print(f"  Model {m_idx+1} best val acc: {best_val_acc:.4f}")

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def predict(self, specs_tensor):
        """
        Ensemble prediction for one patient.

        Args:
            specs_tensor (Tensor): Shape (n_segments, H, W).

        Returns:
            probs (np.ndarray): Shape (2,) — [P(Absent), P(Present)].
            uncertainty (float): Std across ensemble members.
        """
        all_probs = []
        for model in self.models:
            model.eval()
            p = self._patient_logits(model, specs_tensor).cpu().numpy()
            all_probs.append(p)
        all_probs = np.concatenate(all_probs, axis=0)   # (N, 2)
        return all_probs.mean(axis=0), float(all_probs.std())

    def load_checkpoints(self, checkpoint_dir):
        """Reload saved weights for all ensemble members."""
        for m_idx, model in enumerate(self.models):
            ckpt = os.path.join(checkpoint_dir, f"model_{m_idx+1}.pth")
            if os.path.exists(ckpt):
                model.load_state_dict(
                    torch.load(ckpt, map_location=self.device))
            else:
                print(f"[WARN] Checkpoint not found: {ckpt}")

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate(ensemble, val_loader, save_fig="confusion_matrix_resnet.png"):
    """
    Run full evaluation on a DataLoader and print classification report.

    Returns:
        dict with accuracy, precision, recall, specificity, f1, auc.
    """
    y_true, y_pred, y_prob = [], [], []

    for batch in val_loader:
        if batch[0] is None:
            continue
        specs_list, labels, _ = batch
        for i, specs in enumerate(specs_list):
            probs, _ = ensemble.predict(specs)
            y_true.append(labels[i].item())
            y_pred.append(int(np.argmax(probs)))
            y_prob.append(float(probs[1]))

    if not y_true:
        print("No predictions made.")
        return {}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "auc":         roc_auc_score(y_true, y_prob),
    }

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS — MurmurGrade-Net ResNet Ensemble")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred,
                                  target_names=["Absent", "Present"],
                                  zero_division=0))
    for k, v in results.items():
        print(f"  {k:<14}: {v:.4f}")

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Absent", "Present"],
                yticklabels=["Absent", "Present"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — MurmurGrade-Net ResNet Ensemble")
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    print(f"\nSaved: {save_fig}")
    return results

# ============================================================================
# INFERENCE HELPERS
# ============================================================================
def predict_single(audio_path, ensemble, preprocessor):
    """
    Predict murmur status for a single WAV file.

    Returns:
        dict with prediction label, confidence, and probabilities.
    """
    specs = preprocessor.extract(audio_path)
    if specs is None:
        return {"error": f"Could not extract features from {audio_path}"}
    specs_t = torch.FloatTensor(specs)
    probs, uncertainty = ensemble.predict(specs_t)
    pred = int(np.argmax(probs))
    return {
        "audio_path":          str(audio_path),
        "prediction":          "Present (Murmur)" if pred == 1 else "Absent (No Murmur)",
        "prediction_class":    pred,
        "confidence":          float(np.max(probs)),
        "probability_absent":  float(probs[0]),
        "probability_present": float(probs[1]),
        "uncertainty":         uncertainty,
    }


def batch_predict(audio_paths, ensemble, preprocessor):
    """Predict murmur status for a list of WAV files. Returns a DataFrame."""
    return pd.DataFrame([predict_single(p, ensemble, preprocessor)
                         for p in audio_paths])

# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="MurmurGrade-Net ResNet Ensemble")
    p.add_argument("--data_dir",       required=True,
                   help="CirCor training_data directory (contains .txt + .wav files)")
    p.add_argument("--mode",           default="train",
                   choices=["train", "evaluate", "predict"])
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--num_models",     type=int,   default=15,
                   help="Number of ensemble members (default: 15)")
    p.add_argument("--epochs",         type=int,   default=18)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--test_size",      type=float, default=0.2)
    p.add_argument("--seed",           type=int,   default=42)
    # for --mode predict only
    p.add_argument("--audio_path",     default=None,
                   help="Single WAV file to predict (used with --mode predict)")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    preprocessor = PCGPreprocessor()
    ensemble = MurmurEnsemble(num_models=args.num_models, device=device)

    if args.mode == "predict":
        if not args.audio_path:
            raise ValueError("--audio_path is required for --mode predict")
        ensemble.load_checkpoints(args.checkpoint_dir)
        result = predict_single(args.audio_path, ensemble, preprocessor)
        for k, v in result.items():
            print(f"  {k}: {v}")
        return

    # Build dataset and loaders
    dataset = CircorPCGDataset(args.data_dir, preprocessor=preprocessor)
    if len(dataset) == 0:
        raise RuntimeError("No data loaded. Check --data_dir path.")

    indices = np.arange(len(dataset))
    labels  = np.array([d["label"] for d in dataset.data])
    train_idx, val_idx = train_test_split(indices, test_size=args.test_size,
                                           stratify=labels,
                                           random_state=args.seed)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    if args.mode == "train":
        ensemble.train(train_loader, val_loader,
                       epochs=args.epochs, lr=args.lr,
                       checkpoint_dir=args.checkpoint_dir)
        ensemble.load_checkpoints(args.checkpoint_dir)
        evaluate(ensemble, val_loader)

    elif args.mode == "evaluate":
        ensemble.load_checkpoints(args.checkpoint_dir)
        evaluate(ensemble, val_loader)


if __name__ == "__main__":
    main()
