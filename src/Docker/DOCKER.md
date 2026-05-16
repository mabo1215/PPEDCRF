# Docker Usage Guide

## Build the Docker Image

```bash
docker build -t ppedcrf:latest .
```

或指定特定的 CUDA 版本（如果需要）：

```bash
docker build --build-arg DEBIAN_FRONTEND=noninteractive -t ppedcrf:latest .
```

## Run the Container

### 基础用法 - 查看帮助信息

```bash
docker run --rm --gpus all ppedcrf:latest src/main.py --help
```

### 训练模型

```bash
docker run --rm --gpus all \
  -v $(pwd)/src/data/driving:/workspace/src/data/driving \
  -v $(pwd)/src/outputs:/workspace/src/outputs \
  ppedcrf:latest src/main.py --config src/config/config.yaml train
```

或带自定义参数：

```bash
docker run --rm --gpus all \
  -v $(pwd)/src/data/driving:/workspace/src/data/driving \
  -v $(pwd)/src/outputs:/workspace/src/outputs \
  ppedcrf:latest src/main.py --config src/config/config.yaml train \
    --epochs 20 --batch_size 4 --lr 1e-4
```

### 执行攻击评估

```bash
docker run --rm --gpus all \
  -v $(pwd)/src/data/driving:/workspace/src/data/driving \
  -v $(pwd)/src/outputs:/workspace/src/outputs \
  ppedcrf:latest src/main.py --config src/config/config.yaml attack
```

### 隐私保护

```bash
docker run --rm --gpus all \
  -v $(pwd)/src/data/driving:/workspace/src/data/driving \
  -v $(pwd)/src/outputs:/workspace/src/outputs \
  ppedcrf:latest src/main.py --config src/config/config.yaml protect \
    --checkpoint src/outputs/sensnet_final.pt
```

或使用预训练模型：

```bash
docker run --rm --gpus all \
  -v $(pwd)/src/data/driving:/workspace/src/data/driving \
  -v $(pwd)/src/outputs:/workspace/src/outputs \
  ppedcrf:latest src/main.py --config src/config/config.yaml protect \
    --checkpoint mabo1215/ppedcrf-sensnet
```

### 运行辅助脚本

```bash
docker run --rm --gpus all \
  -v $(pwd)/src/data/driving:/workspace/src/data/driving \
  -v $(pwd)/src/outputs:/workspace/src/outputs \
  ppedcrf:latest src/scripts/run_attack_multiseed.py --config src/config/config.yaml
```

```bash
docker run --rm --gpus all \
  -v $(pwd)/src/outputs:/workspace/src/outputs \
  ppedcrf:latest src/scripts/compute_quality_table.py --config src/config/config.yaml
```

## 关键参数说明

- `--rm`：容器退出后自动删除容器
- `--gpus all`：启用所有 GPU（需要安装 nvidia-docker）
- `-v host_path:container_path`：挂载本地目录到容器内
- `$(pwd)`：当前工作目录（在 PowerShell 中使用 `${pwd}`）

## 数据准备

在运行容器前，确保数据结构如下：

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
    clip_0001/
      ...
```

或者使用视频格式：

```
src/data/driving/
  train/
    clip_0001.mp4
    clip_0002.mp4
  val/
    ...
```

## GPU 支持

需要安装 NVIDIA Docker 支持：

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 交互式开发

进入容器交互式 Shell：

```bash
docker run --rm --gpus all -it \
  -v $(pwd)/src:/workspace/src \
  ppedcrf:latest /bin/bash
```

## 注意事项

1. 第一次构建可能需要较长时间（需要下载 CUDA 基础镜像和所有依赖）
2. 使用 `--gpus all` 需要 Docker 支持 GPU（nvidia-docker2）
3. 数据路径需要正确挂载，否则容器无法访问训练数据
4. 如果模型很大，输出目录可能需要足够的磁盘空间
