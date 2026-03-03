# PPEDCRF — Research Codebase

PPEDCRF implements a research scaffold for privacy-preserving video perturbation using
dynamic Conditional Random Fields (CRF) and normalized control penalty (NCP).  
**Paper:** [arXiv:2603.01593](https://arxiv.org/abs/2603.01593) | [Hugging Face](https://huggingface.co/papers/2603.01593)  
**Pre-trained model:** [mabo1215/ppedcrf-sensnet](https://huggingface.co/mabo1215/ppedcrf-sensnet) on the Hub.

## Key features
- Dynamic-CRF refinement for temporal + spatial smoothing of sensitive background regions
- NCP (Normalized Control Penalty) for allocating perturbation intensity
- Background-targeted noise injection (Gaussian / Wiener-style)
- Retrieval-attack evaluation (Top-K / Recall@K) to measure privacy leakage
- Utility metrics (PSNR / SSIM) to quantify visual quality degradation

Project layout
```text
.
├─ main.py
├─ run_train.py
├─ requirements.txt
├─ config/
│  └─ config.yaml
├─ datasets/
│  └─ driving_clip_dataset.py
├─ eval/
│  ├─ metrics.py
│  └─ retrieval_attack.py
├─ models/
│  └─ dynamic_crf.py
├─ privacy/
│  ├─ ncp.py
│  └─ noise_injector.py
├─ utils/
│  └─ config.py
└─ .cursor/
   ├─ .cursorignore
   └─ rules/
      ├─ core.mdc
      ├─ python-style.mdc
      └─ experiments.mdc
```

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
      000001.jpg
      000002.jpg
      ...
    val/

- Option B — Video files

  data/driving/
    train/
      clip_0001.mp4
      clip_0002.mp4
    val/

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

- Protect clips (PPEDCRF pipeline) and compute quick utility metric (use local checkpoint or the pre-trained model from the Hub):

```bash
# Local checkpoint
python main.py --config config/config.yaml protect --checkpoint outputs/sensnet_final.pt
# Pre-trained model on Hugging Face
python main.py --config config/config.yaml protect --checkpoint mabo1215/ppedcrf-sensnet
```

Outputs
- Checkpoints are saved to `train.out_dir` (default `outputs/`)
- Final sensitive-region predictor (example): `outputs/sensnet_final.pt`
- **Pre-trained on Hub:** [mabo1215/ppedcrf-sensnet](https://huggingface.co/mabo1215/ppedcrf-sensnet)

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

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## Citation

If you use this code or the paper in your work, please cite:

**PPEDCRF: Privacy-Preserving Enhanced Dynamic CRF for Location-Privacy Protection for Sequence Videos with Minimal Detection Degradation**  
Bo Ma, Jinsong Wu, Weiqi Yan, Catherine Shi, Minh Nguyen. *arXiv:2603.01593*, 2026.

- **arXiv:** [https://arxiv.org/abs/2603.01593](https://arxiv.org/abs/2603.01593)

```bibtex
@article{ma2026ppedcrf,
  title={PPEDCRF: Privacy-Preserving Enhanced Dynamic CRF for Location-Privacy Protection for Sequence Videos with Minimal Detection Degradation},
  author={Ma, Bo and Wu, Jinsong and Yan, Weiqi and Shi, Catherine and Nguyen, Minh},
  journal={arXiv preprint arXiv:2603.01593},
  year={2026}
}
```