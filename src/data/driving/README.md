# Driving Clip Dataset Layout

Place the dataset under this directory using one of the following layouts. The
default configuration expects `data.root: "src/data/driving"` in
`src/config/config.yaml`.

## Option A: frame folders

```
src/data/driving/
  train/
    clip_0001/
      000001.jpg
      000002.jpg
      ...
    clip_0002/
      ...
  val/
    clip_xxxx/
      ...
```

## Option B: video files

```
src/data/driving/
  train/
    clip_0001.mp4
    clip_0002.mp4
    ...
  val/
    ...
```

- Running `attack` uses `train` as the gallery split and `val` as the query split.
- Put at least a few clips into both `train/` and `val/` before running:
  `python src/main.py --config src/config/config.yaml attack`
