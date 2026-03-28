# Updating paper numbers from scripts

The paper (`paper/main.tex`) uses placeholders that are filled by running the following scripts with your dataset and environment.

## 1. NCP privacy-to-privacy nearest-neighbor distance (d0, d1)

**Placeholders:** `\texttt{<d0>}`, `\texttt{<d1>}` (Section IV, Fig. labelMOT paragraph)

**Script:** `compute_privacy_neighbor_distances.py`

```bash
# Install deps if needed: opencv-python, torch, torchvision. Config default backbone is yolov11n (needs ultralytics).
# To avoid ultralytics/cv2 in embedder, use resnet18:
python src/scripts/compute_privacy_neighbor_distances.py --backbone resnet18 --max_clips 20

# Update paper (paper/main.tex) with computed d0, d1:
python src/scripts/compute_privacy_neighbor_distances.py --backbone resnet18 --update-tex
```

Optional: `--tex paper/main.tex` (default), `--N_max`, `--N_min` for class counts.

## 2. Top-k retrieval accuracy mean ± std (R@1, R@5, R@10)

**Placeholders:** `\texttt{<R1>}`, `\texttt{<R5>}`, `\texttt{<R10>}`, `\texttt{<R1std>}`, `\texttt{<R5std>}`, `\texttt{<R10std>}` (Section IV, Reproducibility / Threat model)

**Script:** `run_attack_multiseed.py`

Runs the retrieval attack on *protected* query frames under 3 noise seeds and reports mean ± std.

```bash
# Run with default config (needs data root, checkpoint, and dataset with train/val splits):
python src/scripts/run_attack_multiseed.py

# If your config uses a backbone that requires ultralytics (e.g. yolov11n), use resnet18:
python src/scripts/run_attack_multiseed.py --backbone resnet18

# Write mean ± std into paper/main.tex:
python src/scripts/run_attack_multiseed.py --update-tex
```

Optional: `--seeds 1234 1235 1236`, `--tex paper/main.tex`, `--max_query`, `--max_gallery`.

## 3. Figure path

The architecture figure path in `paper/main.tex` is set to `images/architecture_of_solution.png`.  
A copy of the original file was created as `paper/images/architecture_of_solution.png`.  
If you only have `artecture of solution.png`, copy it to `architecture_of_solution.png` in the same folder.
