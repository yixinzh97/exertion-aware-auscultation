# Exertion-Aware AI Auscultation

**Exertion-aware AI auscultation for reducing false pathological murmur alerts after exercise.**  
This repository supports the paper:

> **Exploring Exertion-Aware AI Auscultation: Addressing Exercise-Induced Murmurs to Reduce False Pathological Alerts**  
> Yixin Zhang, Shaaf Ahmad, Faisal F. Syed, and Jingping Nie  
> University of North Carolina at Chapel Hill

Most murmur detection systems are trained on **resting-condition PCG recordings** and may mistake exercise-induced physiological acoustic changes for pathological murmurs. This project studies that failure mode using a clinician-annotated post-exercise PCG subset and introduces an **Exertion-Aware Dual Bayesian ResNet** that explicitly separates:

- **NM**: no murmur
- **EM**: exercise-induced physiological murmur
- **PM**: pathological murmur

The best feature configuration, **Mel + PSD + RMS**, achieves **91.46% overall accuracy** and **77.70% EM recall** in 5-fold cross-validation.

---

## Highlights

- Clinician-annotated **post-exercise PCG subset** with 160 recordings collected immediately after treadmill running.
- Verified non-pathological exercise-induced murmurs: **108 NM** and **52 EM** recordings.
- Systematic evaluation of resting-trained baselines under post-exercise conditions.
- Exertion-aware training strategy combining **Post-Exercise DB** with **CirCor DB** to retain pathological sensitivity while learning EM-specific patterns.
- Dual Bayesian ResNet backbone with Monte Carlo Dropout and segment-level Bayesian inference.
- Feature augmentation using **log-Mel spectrograms**, **RMS energy**, **MFCCs**, and **PSD energy**.

---

## Key Results

### 5-Fold Cross-Validation: Feature Combinations

Values are reported as mean ± half-width of the 95% confidence interval.

| Feature Config | Overall Acc. | EM Recall | PM Recall |
|---|---:|---:|---:|
| Mel | 0.8856 ± 0.0266 | 0.6511 ± 0.1933 | 0.7178 ± 0.1056 |
| Mel + RMS | 0.8910 ± 0.0049 | 0.4857 ± 0.1680 | 0.6366 ± 0.0481 |
| Mel + PSD | 0.9039 ± 0.0066 | 0.5429 ± 0.1633 | 0.6676 ± 0.0387 |
| Mel + MFCC | 0.8877 ± 0.0131 | 0.7429 ± 0.1048 | 0.7465 ± 0.0231 |
| **Mel + PSD + RMS** | **0.9146 ± 0.0118** | **0.7770 ± 0.1067** | **0.6799 ± 0.0590** |
| Mel + PSD + RMS + MFCC | 0.9120 ± 0.0098 | 0.7080 ± 0.2470 | 0.6830 ± 0.1385 |

The final model uses **Mel + PSD + RMS**, prioritizing improved EM recall while maintaining acceptable PM discrimination.

---

## Repository Structure

```text
exertion-aware-auscultation/
├── data/
│   └── annotations/
│       ├── post_exercise_labels.csv      # Clinician-annotated EM/NM labels
│       └── README_data.md                # Instructions for obtaining raw PCG recordings
│   └── data_all/                         # PM / EM / NM data for 3-class model
├── models/
│   └── exertion_aware_dual_bayesian.py   # Proposed model
├── baselines/
│   ├── resnet.py                         # ResNet murmur grading baseline
│   └── lstm.py                           # HM-Detect LSTM baseline
├── configs/
│   └── default.yaml
├── experiments/
│   ├── train.py
│   ├── evaluate.py
│   └── cross_validation.py
└── requirements.txt
```

---

## Datasets

### Post-Exercise DB

The post-exercise subset is derived from the **Multi-Modal Post-Exercise Dataset** and contains **160 PCG recordings** collected immediately after treadmill running at varying exertion levels.

Collection details:

- Recording device: **3M Littmann CORE digital stethoscope**
- Recording site: **mitral valve position**
- Exercise protocol: treadmill running with **Rating of Perceived Exertion (RPE) levels 1–5**
- Quality control: recordings with severe signal corruption, sensor instability, or dominant non-cardiac artifacts were excluded
- Retained signals include post-exercise physiological signatures such as elevated heart rate and increased amplitude variability

Clinician annotation:

- Annotated by a **board-certified cardiologist** through auditory review
- Supported by physician waveform cross-reference
- Ambiguous recordings were excluded rather than force-labeled
- All post-exercise murmurs were verified as **non-pathological physiological flow murmurs**

Label distribution:

| Label | Count | Description |
|---|---:|---|
| NM | 108 | No murmur |
| EM | 52 | Exercise-induced physiological murmur, verified non-pathological |

Annotation file:

```text
data/annotations/post_exercise_labels.csv
```

Expected columns:

```text
recording_id, rpe_level, label, heart_rate_bpm, norm_peak_amplitude
```

### CirCor DB

The **CirCor DigiScope Phonocardiogram Dataset** contains **3,164 resting-condition PCG recordings** with detailed murmur annotations. In this project, CirCor DB is used for:

- Baseline pretraining and replication
- Joint training with Post-Exercise DB for the proposed exertion-aware model
- Retaining PM sensitivity while learning to distinguish EM from PM

Dataset link:

```text
https://physionet.org/content/circor-heart-sound/1.0.3/
```

---

## Baseline Evaluation Under Post-Exercise Conditions

Resting-trained models were evaluated directly on the clinician-annotated Post-Exercise DB. All baseline outputs were standardized into a common binary decision: **NM vs. PM**.

| Baseline Model | NM Accuracy | EM → PM Misclassification |
|---|---:|---:|
| ResNet | 58.33% | 61.54% |
| LSTM / HM-Detect | 47.22% | 36.54% |
| Dual Bayesian ResNet | 54.63% | 88.46% |

These results show that models trained only on resting-condition data frequently overpredict pathology when applied to post-exercise PCG recordings.

A commercial automated detection system also showed disagreement with expert ground truth. On the post-exercise cohort, it produced false pathological alerts for **10.3%** of cardiologist-verified non-pathological recordings and returned **Unknown** labels for **15.2%** of recordings.

---

## Exertion-Aware Dual Bayesian ResNet

The proposed model extends the Dual Bayesian ResNet framework by explicitly modeling the three-class taxonomy **NM / EM / PM**.

### Architecture

The model uses two Bayesian ResNet-18 branches:

1. **Branch A:** separates PM from non-pathological states, i.e., PM vs. NM + EM
2. **Branch B:** separates EM from NM

Segment-level posterior probabilities are aggregated at the recording level by averaging predictions across overlapping segments. Monte Carlo Dropout is used for uncertainty-aware inference.

### Signal Preprocessing

- Resample PCG recordings to **4 kHz**
- Segment into **4-second windows**
- Use **1-second stride**, corresponding to 75% overlap
- Compute STFT with:
  - Hann window
  - `NFFT = 0.025 × SR`
  - `HOP = 0.010 × SR`
  - numerical stability constant `epsilon = 1e-10`
- Extract 64-channel log-Mel spectrogram over **10–2000 Hz**

### Feature Augmentation

The final feature representation concatenates:

- 64-channel log-Mel spectrogram: `[64, T]`
- RMS energy: `[1, T]`
- PSD energy: `[1, T]`

For full feature experiments, MFCCs are also evaluated:

- MFCCs: `[13, T]`

All engineered features are min–max normalized within each segment and linearly interpolated to align with the Mel time axis.

---

## Computational Cost

The Dual Bayesian ResNet uses two ResNet-18 Bayesian branches.

| Metric | Value |
|---|---:|
| Parameters per branch | 11.18M |
| Total parameters | 22.36M |
| FLOPs per 4-second segment | 1.14 GFLOPs |
| Deterministic forward pass on Apple M2 | 2.09 ms per segment |
| 10-pass MC Dropout inference for 10-second recording | ~228 ms |
| FP32 memory footprint | 42.6 MB per branch / 85.3 MB total |
| INT8 estimated memory footprint | ~21.3 MB total |

These results indicate that the model is compatible with real-time auscultation requirements.

---

## Installation

```bash
git clone https://github.com/<your-username>/exertion-aware-auscultation.git
cd exertion-aware-auscultation
pip install -r requirements.txt
```

Requirements:

- Python 3.8+
- PyTorch 1.12+
- librosa
- scikit-learn
- numpy
- pyyaml

---

## Usage

### Train the Exertion-Aware Model

```bash
python experiments/train.py \
  --config configs/default.yaml \
  --circor_path /path/to/circor \
  --postexercise_path /path/to/post_exercise \
  --features mel_psd_rms
```

### Evaluate a Baseline Model

```bash
python experiments/evaluate.py \
  --model dual_bayesian \
  --checkpoint /path/to/pretrained.pt \
  --data_path /path/to/post_exercise \
  --mode baseline
```

### Run 5-Fold Cross-Validation

```bash
python experiments/cross_validation.py \
  --config configs/default.yaml \
  --features mel_psd_rms \
  --n_folds 5
```

---

## Citation

If you use this repository, annotations, or model code, please cite:

```bibtex
@inproceedings{zhang2026exertionaware,
  title     = {Exploring Exertion-Aware AI Auscultation: Addressing Exercise-Induced Murmurs to Reduce False Pathological Alerts},
  author    = {Zhang, Yixin and Ahmad, Shaaf and Syed, Faisal F. and Nie, Jingping},
  booktitle = {Proceedings of IEEE/ACM CHASE},
  year      = {2026},
  note      = {To appear / under review}
}
```

---

## License

Code: [MIT License](LICENSE)  
Annotations: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — free to use with attribution  
Raw PCG recordings: subject to the licensing terms of the Multi-Modal Post-Exercise Dataset

---

## Acknowledgements

This work was supported in part by the NVIDIA Academic Award Grant.

Baseline implementations are adapted from:

- Walker et al., Dual Bayesian ResNet, Computing in Cardiology 2022
- Sivaraman and Xiao, HM-Detect, IEEE SiPS 2024
- Liu et al., automatic murmur grading, ICLR Time Series for Health 2023

Datasets and references:

- CirCor DB: Oliveira et al., IEEE JBHI 2022 / PhysioNet Challenge 2022
- Multi-Modal Post-Exercise Dataset: Nie et al., ACM SenSys 2025

