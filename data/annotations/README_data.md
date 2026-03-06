# Data Instructions

## Post-Exercise PCG Annotations (Open-Sourced)

The clinician-annotated labels for all 160 post-exercise recordings are
provided directly in this repository:

```
data/annotations/post_exercise_labels.csv
```

### Label Distribution

| Label | Count | Description |
|-------|-------|-------------|
| `No Murmur Detected` | 108 | No murmur — verified non-pathological |
| `Murmur Detected` | 52 | Exercise-induced murmur — verified non-pathological |

All murmurs were annotated by a board-certified cardiologist and confirmed
to be **physiological flow murmurs**, not structural cardiac abnormalities.

### Exertion Level Distribution

Recordings span 5 RPE (Rating of Perceived Exertion) levels:

| RPE Level | Total | Murmur Detected | No Murmur Detected |
|-----------|-------|-----------------|-------------------|
| 1 | 14 | 4 | 10 |
| 2 | 42 | 18 | 24 |
| 3 | 52 | 22 | 30 |
| 4 | 48 | 8 | 40 |
| 5 | 4 | 0 | 4 |

### Participant Demographics

| Attribute | Value |
|-----------|-------|
| Total recordings | 160 |
| Age (mean ± std) | 25.7 ± 8.5 years (range: 18–59) |
| Gender | 89 Male / 71 Female |
| Weight (mean ± std) | 63.9 ± 14.2 kg |
| Height (mean ± std) | 171.0 ± 9.6 cm |

### Column Description

| Column | Description |
|--------|-------------|
| `original_filename` | Original WAV filename |
| `pcg_name` | Anonymized recording ID |
| `Murmur_groundtruth` | Cardiologist annotation (`Murmur Detected` / `No Murmur Detected`) |
| `label_from_3Mlittleman` | 3M Littmann CORE commercial algorithm output |
| `label_resnet` | ResNet baseline prediction (pretrained on CirCor DB) |
| `label_lstm` | LSTM baseline prediction (pretrained on CirCor DB) |
| `label_dual_new` | Dual Bayesian ResNet baseline prediction |
| `Exertion` | RPE level at time of recording (1–5) |
| `Age` | Participant age in years |
| `Gender` | Participant gender |
| `Weight (kg)` | Participant weight |
| `Height (cm)` | Participant height |

### Baseline Model Performance on This Dataset

Resting-trained baselines vs. cardiologist ground truth:

| Model | EM Detected | EM Missed | NM Correct | NM False Alert |
|-------|-------------|-----------|------------|----------------|
| 3M Littmann CORE | 35/52 (67.3%) | 3 | 61/108 (56.5%) | 19 (+28 Unknown) |
| ResNet baseline | 20/52 (38.5%) | 32 | 63/108 (58.3%) | 45 |
| LSTM baseline | 33/52 (63.5%) | 19 | 51/108 (47.2%) | 57 |
| Dual Bayesian baseline | 6/52 (11.5%) | 5 | 59/108 (54.6%) | 49 (+1 Unknown) |

These results illustrate the systematic failure of resting-trained models
under post-exercise conditions (see paper Section IV for full analysis).

---

## Raw PCG Recordings

The raw `.wav` files are part of the **Multi-Modal Post-Exercise Dataset**:

> Nie J., Fan Y., Zhao M., Wan R., Xuan Z., Preindl M., and Jiang X.
> "Multi-modal dataset across exertion levels: Capturing post-exercise
> speech, breathing, and phonocardiogram."
> *Proceedings of ACM SenSys*, 2025, pp. 297–304.

**Access:** Please contact the dataset authors to request access.

**Recording details:**
- Device: 3M Littmann CORE digital stethoscope
- Position: Mitral valve
- Protocol: Recorded immediately after treadmill running at RPE 1–5
- Total: 160 recordings (after quality control)

---

## CirCor DigiScope Dataset

Baseline pretraining uses the CirCor DigiScope Phonocardiogram Dataset:

> Oliveira J.H. et al. "The CirCor DigiScope Phonocardiogram Dataset."
> *IEEE Journal of Biomedical and Health Informatics*, 2022.

**Download:** https://physionet.org/content/circor-heart-sound/1.0.3/

---

## Matching Annotations to Raw Files

Once you have the raw recordings, match them using `pcg_name`:

```python
import pandas as pd

labels = pd.read_csv("data/annotations/post_exercise_labels.csv")
# pcg_name matches the .wav filename (without extension)
# e.g., pcg_name "d31_P01_6_0" → "d31_P01_6_0.wav"
print(labels[['pcg_name', 'Murmur_groundtruth', 'Exertion']].head())
```
