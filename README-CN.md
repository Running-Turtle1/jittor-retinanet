# jittor-retinanet


本项目基于 [Jittor 框架](https://github.com/Jittor/jittor) 复现了经典的目标检测模型 [RetinaNet](https://arxiv.org/pdf/1708.02002v2.pdf)，对齐并评估了 PyTorch 与 Jittor 两种实现方式在 **COCO 2017** 数据集上的性能与准确率表现。


### 项目内容


- 基于 [PyTorch 实现](https://github.com/yhenon/pytorch-retinanet)，测试了在使用 **预训练 ResNet50 作为 Backbone** 时的检测性能；
- 使用 **Jittor 框架** 对齐 PyTorch 实现，并在相同配置下进行训练和评估；
- 对 **两种版本（PyTorch / Jittor）** 的训练性能、检测精度（mAP）、收敛速度和损失变化进行了系统对比分析。

### 项目结构

```wiki

```

### 环境配置

- Pytorch

```bash
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests
```

- Jittor

```bash
docker pull jittor/jittor-cuda:11.1-16.04
```

其他安装方式见 [Jittor](https://github.com/Jittor/jittor) 官网。

### 使用方法

#### pytorch-retinanet

- 数据准备 - COCO 2017 数据集

```bash
python prepare_data.py
```

- pytorch 训练

```bash
python train.py --dataset coco --coco_path ./coco --depth 50  --epochs 35 --sample_size 10000 --batch_size 32
```

- 验证

```bash
python coco_validation.py --coco_path ./coco --model <your_model_path>.pt
```

#### jittor-retinanet

### 对齐验证

### 可视化结果

### 模型权重

我们提供了训练完成的 RetinaNet 模型权重：

| 模型               | 下载链接 |
| ------------------ | -------- |
| Pytorch + ResNet18 |          |
| Pytorch + ResNet34 |          |
| Pytorch + ResNet50 |          |
|Jittor + ResNet18||
|Jittor + ResNet34||
|Jittor + ResNet50||

