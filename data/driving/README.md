# 驾驶片段数据目录

把数据按以下结构放在本目录下（与 `config/config.yaml` 中 `data.root: "data/driving"` 对应）：

## 方式 A：按帧的图片文件夹

```
data/driving/
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

## 方式 B：视频文件

```
data/driving/
  train/
    clip_0001.mp4
    clip_0002.mp4
    ...
  val/
    ...
```

- 运行 `attack` 会用到 `train`（gallery）和 `val`（query）。
- 至少放一些片段到 `train/` 和 `val/` 后再执行：
  `python main.py --config config\config.yaml attack`
