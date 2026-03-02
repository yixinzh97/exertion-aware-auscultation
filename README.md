# exertion-aware-auscultation
Exertion-aware AI auscultation for reducing false pathological murmur alerts after exercise. Clinician-annotated post-exercise PCG dataset (n=160) + Dual Bayesian ResNet with RMS/PSD feature augmentation. 91.46% accuracy, 77.70% EM recall.

# Exertion-Aware AI Auscultation

> **Addressing Exercise-Induced Murmurs to Reduce False Pathological Alerts**  
> *[Short Paper — Anonymous Submission]*

---

## Overview

Most AI murmur detection systems are trained on **resting-condition** recordings and fail under post-exercise conditions — flagging physiological flow murmurs as pathological. This repository addresses that gap by:

1. Providing **clinician-annotated post-exercise PCG labels** (n=160) with verified exercise-induced murmur (EM) ground truth
2. Introducing an **Exertion-Aware Dual Bayesian ResNet** that explicitly separates NM / EM / PM
3. Providing **reproducible baseline implementations** (ResNet, LSTM, Dual Bayesian ResNet) evaluated on post-exercise conditions

**Key results (5-fold cross-validation):**

| Feature Config | Overall Acc. | EM Recall | PM Recall |
|---|---|---|---|
| Mel only | 88.56% | 65.11% | 71.78% |
| Mel + MFCC | 88.77% | 74.29% | 74.65% |
| **Mel + PSD + RMS** | **91.46%** | **77.70%** | **67.99%** |
| Mel + RMS + MFCC + PSD | 91.20% | 70.79% | 68.26% |

---

## Repository Structure

```
exertion-aware-auscultation/
├── data/
│   └── annotations/
│       ├── post_exercise_labels.csv    # Clinician-annotated EM/NM labels (open-sourced)
│       └── README_data.md              # Instructions for obtaining raw PCG recordings
├── models/
│   ├── exertion_aware_dual_bayesian.py # Proposed model (main contribution)
│   └── feature_extraction.py           # RMS, PSD, MFCC feature augmentation
├── baselines/
│   ├── resnet_murmur.py                # ResNet murmur grading (Liu et al.)
│   ├── lstm_hm_detect.py               # HM-Detect LSTM (Sivaraman & Xiao)
│   └── dual_bayesian_resnet.py         # Dual Bayesian ResNet (Walker et al.)
├── experiments/
│   ├── train.py                        # Training entry point
│   ├── evaluate.py                     # Evaluation + confusion matrix
│   └── cross_validation.py             # 5-fold CV protocol
├── configs/
│   └── default.yaml                    # Hyperparameters and paths
└── requirements.txt
```

---

## Dataset

### Post-Exercise PCG Annotations (Open-Sourced)

We open-source the **clinician annotations** for our post-exercise PCG subset.  
Labels are in `data/annotations/post_exercise_labels.csv`:

```
recording_id, rpe_level, label, heart_rate_bpm, norm_peak_amplitude
```

| Label | Count | Description |
|---|---|---|
| NM | 108 | No murmur |
| EM | 52 | Exercise-induced physiological murmur (verified non-pathological) |

All murmurs were annotated by a **board-certified cardiologist** and confirmed non-pathological — representing physiological flow murmurs induced by exertion, not structural abnormalities.

### Obtaining the Raw PCG Recordings

Raw recordings are part of the **Multi-Modal Post-Exercise Dataset** and are available via:
> Nie et al., "Multi-modal dataset across exertion levels: Capturing post-exercise speech, breathing, and phonocardiogram," *ACM SenSys 2025*.

Hardware: 3M Littmann CORE digital stethoscope, mitral valve position, RPE levels 1–5.

### CirCor DB (External)

Baseline pretraining and joint training use the **CirCor DigiScope Phonocardiogram Dataset**:  
→ https://physionet.org/content/circor-heart-sound/1.0.3/

---

## Installation

```bash
git clone https://github.com/<your-username>/exertion-aware-auscultation.git
cd exertion-aware-auscultation
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch 1.12+, librosa, scikit-learn, numpy, pyyaml

---

## Usage

### Training the Exertion-Aware Model

```bash
python experiments/train.py \
  --config configs/default.yaml \
  --circor_path /path/to/circor \
  --postexercise_path /path/to/post_exercise \
  --features mel_psd_rms
```

### Running Baseline Evaluation

```bash
# Evaluate pretrained baselines on Post-Exercise DB
python experiments/evaluate.py \
  --model dual_bayesian \
  --checkpoint /path/to/pretrained.pt \
  --data_path /path/to/post_exercise \
  --mode baseline
```

### 5-Fold Cross-Validation

```bash
python experiments/cross_validation.py \
  --config configs/default.yaml \
  --features mel_psd_rms \
  --n_folds 5
```

---

## Model Architecture

The **Exertion-Aware Dual Bayesian ResNet** extends the DBRes framework (Walker et al., 2022) with:

- **3-class taxonomy**: NM / EM / PM (vs. original binary NM vs. PM)
- **Dual binary decomposition**: ResNet-A (PM vs. NM+EM) + ResNet-B (EM vs. NM)
- **Monte Carlo Dropout** for uncertainty-aware inference
- **Feature augmentation**: log-Mel spectrogram + RMS energy + PSD (fused [79, T] input)

**Signal preprocessing:**
- Resampled to 4 kHz
- Segmented into 4s windows, 1s stride (75% overlap)
- STFT: Hann window, NFFT = 0.025 × SR, hop = 0.010 × SR
- 64-channel log-Mel spectrogram over [10, 2000] Hz

---

## Baseline Failure Analysis

Pretrained baselines (CirCor DB only) evaluated on Post-Exercise DB show systematic failure:

| Model | NM Accuracy | EM → PM Misclassification |
|---|---|---|
| ResNet | 58.33% | 61.54% |
| LSTM | 47.22% | 36.54% |
| Dual Bayesian ResNet | 54.63% | **88.46%** |

> Exercise-induced acoustic changes (elevated HR, increased amplitude variability, respiratory overlap) push all resting-trained models toward false pathological predictions — regardless of architecture.

---

## Citation

If you use our annotations or code, please cite:

```bibtex
@inproceedings{anonymous2025exertion,
  title     = {Exploring Exertion-Aware AI Auscultation: Addressing Exercise-Induced Murmurs to Reduce False Pathological Alerts},
  author    = {Anonymous Authors},
  booktitle = {[Venue TBD]},
  year      = {2025}
}
```

---

## License

Code: [MIT License](LICENSE)  
Annotations: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use with attribution  
Raw PCG recordings: subject to the licensing terms of the Multi-Modal Post-Exercise Dataset

---

## Acknowledgements

Baseline implementations are adapted from:
- Walker et al., [Dual Bayesian ResNet](https://github.com/), CinC 2022
- Sivaraman & Xiao, HM-Detect, IEEE SiPS 2024
- Liu et al., Murmur Grading, ICLR Time Series for Health 2023

CirCor DB: Oliveira et al., IEEE JBHI 2022 / PhysioNet Challenge 2022
