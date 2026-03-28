# PPEDCRF Research Codebase

PPEDCRF implements a research scaffold for privacy-preserving video perturbation using
dynamic Conditional Random Fields (CRF) and normalized control penalty (NCP).

Paper: [arXiv:2603.01593](https://arxiv.org/abs/2603.01593)  
Pre-trained model: [mabo1215/ppedcrf-sensnet](https://huggingface.co/mabo1215/ppedcrf-sensnet)

## Layout

```text
.
|-- README.md
|-- docs/
|-- paper/
`-- src/
    |-- config/
    |   `-- config.yaml
    |-- data/
    |   `-- driving/
    |       `-- README.md
    |-- datasets/
    |   `-- driving_clip_dataset.py
    |-- eval/
    |   |-- metrics.py
    |   `-- retrieval_attack.py
    |-- models/
    |   `-- dynamic_crf.py
    |-- privacy/
    |   |-- NCP.py
    |   `-- noise_injector.py
    |-- scripts/
    |   |-- compute_privacy_neighbor_distances.py
    |   |-- compute_quality_table.py
    |   |-- run_attack_multiseed.py
    |   |-- split_train_val.py
    |   `-- update_tex_placeholders.py
    |-- utils/
    |   `-- config.py
    |-- main.py
    |-- requirements.txt
    |-- run_eval.py
    `-- run_train.py
```

## Setup

Python 3.8+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r src/requirements.txt
```

## Data Format

The default config points to `src/data/driving`, and the repository accepts two clip formats.

Option A: frame folders

```text
src/data/driving/
  train/
    clip_0001/
      000001.jpg
      000002.jpg
  val/
```

Option B: video files

```text
src/data/driving/
  train/
    clip_0001.mp4
    clip_0002.mp4
  val/
```

If your dataset lives elsewhere, override `data.root` in `src/config/config.yaml` or pass `--data_root`.

## Usage

Default configuration: `src/config/config.yaml`

```bash
python src/main.py --config src/config/config.yaml train
python src/main.py --config src/config/config.yaml train --epochs 20 --batch_size 4 --lr 1e-4
python src/main.py --config src/config/config.yaml attack
python src/main.py --config src/config/config.yaml protect --checkpoint src/outputs/sensnet_final.pt
python src/main.py --config src/config/config.yaml protect --checkpoint mabo1215/ppedcrf-sensnet
```

Useful helper scripts:

```bash
python src/scripts/run_attack_multiseed.py --config src/config/config.yaml
python src/scripts/compute_quality_table.py --config src/config/config.yaml
python src/scripts/compute_privacy_neighbor_distances.py --config src/config/config.yaml
```

## Outputs

- Training checkpoints are saved to `src/outputs/` by default.
- The final sensitive-region predictor is written to `src/outputs/sensnet_final.pt`.

If you have GT masks, set `train.mask_root` in `src/config/config.yaml`, for example:

```yaml
train:
  mask_root: "src/data/masks"
```

Expected mask layout:

```text
src/data/masks/
  train/
    clip_0001/
      000001.png
      000002.png
  val/
  test/
```

## Notes

- The training loop uses a placeholder mask if `train.mask_root` is null.
- The codebase is intentionally modular, so you can swap the sensitive-region model, embedder, or threat model.
- `paper/` stays at the repository root because it is a Git submodule.

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
