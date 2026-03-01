# PPEDCRF — Research Codebase

PPEDCRF implements a research scaffold for privacy-preserving video perturbation using
dynamic Conditional Random Fields (CRF) and normalized control penalty (NCP).

Key features
- Dynamic-CRF refinement for temporal + spatial smoothing of sensitive background regions
- NCP (Normalized Control Penalty) for allocating perturbation intensity
- Background-targeted noise injection (Gaussian / Wiener-style)
- Retrieval-attack evaluation (Top-K / Recall@K) to measure privacy leakage
- Utility metrics (PSNR / SSIM) to quantify visual quality degradation

Project layout
```
.
├─ main.py
├─ run_train.py
├─ requirements.txt
├─ config/
│  └─ config.yaml
├─ ppedcrf/
│  ├─ datasets/
│  │  └─ driving_clip_dataset.py
│  ├─ eval/
│  │  ├─ metrics.py
│  │  └─ retrieval_attack.py
│  ├─ models/
│  │  └─ dynamic_crf.py
│  ├─ privacy/
│  │  ├─ ncp.py
│  │  └─ noise_injector.py
│  └─ utils/
│     └─ config.py
└─ .cursor/
  ├─ .cursorignore
  └─ rules/
    ├─ core.mdc
  ├─ python-style.mdc
  └─ experiments.mdc

Prerequisites
- Python 3.8+ recommended
- pip (for installing requirements)
- Optional: CUDA-enabled GPU for faster training

Environment setup
1) Create and activate a virtual environment

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

Dataset format
The repository accepts two clip formats (configured via `data.root` in `config/config.yaml`):

- Option A — Frame folders

  data/driving/
    train/
      clip_0001/
        000001.jpg
    000002.jpg
    ...
  val/
    test/

- Option B — Video files

  data/driving/
    train/
      clip_0001.mp4
      clip_0002.mp4
  val/
  test/

The loader automatically detects frame folders or video files.

Configuration
Default configuration: `config/config.yaml`.
All CLI commands accept `--config` to load a YAML file and allow overriding common settings via flags.

Usage examples
- Train the sensitive-region predictor

```bash
python main.py --config config/config.yaml train
```

- Train with overrides

```bash
python main.py --config config/config.yaml train --epochs 20 --batch_size 4 --lr 1e-4
```

- Run retrieval attack evaluation

```bash
python main.py --config config/config.yaml attack
```

- Protect clips (PPEDCRF pipeline) and compute quick utility metric

```bash
python main.py --config config/config.yaml protect --checkpoint outputs/sensnet_final.pt
```

Outputs
- Checkpoints are saved to `train.out_dir` (default `outputs/`)
- Final sensitive-region predictor (example): `outputs/sensnet_final.pt`

Pipeline steps performed by `protect`
- Sensitive-region inference (unary logits)
- Dynamic-CRF refinement
- NCP allocation
- Background-targeted noise injection
- Reports PSNR (currently computed on the first frame as a quick sanity metric)

Ground-truth masks (optional)
If you have GT masks, set in `config/config.yaml`:

```yaml
train:
  mask_root: "data/masks"
```

Expected mask layout

```
data/masks/
  train/
    clip_0001/
      000001.png
      000002.png
      ...
  val/
  test/
```

Mask format
- Grayscale PNG
- 0 = non-sensitive background
- 255 = sensitive background

Troubleshooting
- OpenCV cannot read videos or is missing:

```bash
pip install opencv-python
# or headless for servers without display
pip install opencv-python-headless
```

- CUDA not available: force CPU device

```bash
python main.py --config config/config.yaml train --device cpu
```

Notes & extension points
- The training loop uses a placeholder mask if `train.mask_root` is null. For meaningful training without GT masks, implement pseudo-labels (for example, saliency from retrieval gradients).
- The codebase is intentionally modular — you can swap the sensitive-region model, embedder, or threat model.

License
This repository is a research scaffold. Add your preferred license (e.g., MIT, Apache-2.0).
# PPEDCRF (Research Codebase)

This repository contains a research implementation scaffold for **PPEDCRF**:
- **Dynamic-CRF refinement** (temporal + spatial smoothing of sensitive background regions)
- **NCP (Normalized Control Penalty)** to allocate perturbation intensity
- **Background-targeted noise injection** (Gaussian / Wiener-style)
- **Retrieval attack evaluation** (Top-K / Recall@K) to measure privacy leakage
- **Utility metrics** (PSNR/SSIM) to quantify visual quality degradation

The code is designed to be:
- **reproducible** (YAML config + optional deterministic seeds),
- **modular** (swap sensitive-region model / embedder / threat model),
- **paper-friendly** (clear tensor shapes and evaluation outputs).

---

## Project Structure
.
├─ main.py
├─ run_train.py
├─ requirements.txt
├─ config/
│ └─ config.yaml
├─ ppedcrf/
│ ├─ datasets/
│ │ └─ driving_clip_dataset.py
│ ├─ eval/
│ │ ├─ metrics.py
│ │ └─ retrieval_attack.py
│ ├─ models/
│ │ └─ dynamic_crf.py
│ ├─ privacy/
│ │ ├─ ncp.py
│ │ └─ noise_injector.py
│ └─ utils/
│ └─ config.py
└─ .cursor/
├─ .cursorignore
└─ rules/
├─ core.mdc
├─ python-style.mdc
└─ experiments.mdc



## Environment Setup

### 1) Create a virtual environment (recommended)



```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2)  Install dependencies
```bash
pip install -r requirements.txt

```

Dataset Format

This repository supports two clip formats under data.root (see config/config.yaml):

Option A: Frame folders
data/driving/
  train/
    clip_0001/
      000001.jpg
      000002.jpg
      ...
  val/
  test/
Option B: Video files
data/driving/
  train/
    clip_0001.mp4
    clip_0002.mp4
  val/
  test/

The loader automatically detects frame folders or video files.

Configuration

Default configuration is stored in:

config/config.yaml

All commands support:

reading configuration from YAML (--config)

overriding common options via CLI flags

Usage
Train (sensitive-region predictor)
python main.py --config config/config.yaml train

Override common settings:

python main.py --config config/config.yaml train --epochs 20 --batch_size 4 --lr 1e-4

Outputs:

checkpoints saved to train.out_dir (default: outputs/)

final model: outputs/sensnet_final.pt

If train.mask_root is null, the provided training loop uses a placeholder mask (for scaffolding only).
For meaningful training without GT masks, implement pseudo labels (e.g., saliency from retrieval gradients).

Retrieval Attack (Top-K / Recall@K)
python main.py --config config/config.yaml attack

Reports:

R@1, R@5, R@10 (Recall@K / Top-K accuracy)

Protect Clips (PPEDCRF) + Utility Metric
python main.py --config config/config.yaml protect --checkpoint outputs/sensnet_final.pt

Runs:

sensitive region inference (unary logits)

Dynamic-CRF refinement

NCP allocation

background-targeted noise injection

Then reports PSNR (currently computed on the first frame as a quick sanity metric).

Ground-Truth Masks (Optional)

If you have GT masks, set in config/config.yaml:

train:
  mask_root: "data/masks"

Expected layout:

data/masks/
  train/
    clip_0001/
      000001.png
      000002.png
      ...
  val/
  test/

Mask format:

grayscale PNG

0 = non-sensitive background

255 = sensitive background

Troubleshooting
OpenCV not installed / cannot read videos
pip install opencv-python
# or headless:
pip install opencv-python-headless
CUDA not available
python main.py --config config/config.yaml train --device cpu
License

This repository is provided as a research scaffold. Add your preferred license here (e.g., MIT, Apache-2.0).