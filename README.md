# jittor-retinanet


本项目基于 [Jittor 框架](https://github.com/Jittor/jittor) 复现经典目标检测模型 [RetinaNet](https://arxiv.org/pdf/1708.02002v2.pdf)，对齐并评估了 PyTorch 与 Jittor 两种实现方式在 **COCO 2017** 数据集上采样部分数据进行训练的性能与精度表现。


### 项目内容


- 基于 [PyTorch 实现](https://github.com/yhenon/pytorch-retinanet)，使用预训练 ResNet50 采样 10,000 张图片进行训练，同时记录训练日志和评估日志；
- 使用 **Jittor 框架** 对齐 PyTorch 实现，并在相同配置下进行训练和评估；
- 对 **两种版本（PyTorch / Jittor）** 的训练性能、检测精度（mAP）、收敛速度和损失变化进行对比分析。

### 项目结构

```wiki

```

### 环境配置

#### Pytorch

```bash
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests
```

#### Jittor

docker安装：

```bash
docker pull jittor/jittor-cuda:11.1-16.04
```

anaconda 安装：

```bash
conda create -n jittor python=3.8
conda activate jittor
conda install pywin32
pip install jittor
# 测试是否安装成功
python -m jittor.test.test_core
python -m jittor.test.test_example
python -m jittor.test.test_cudnn_op
```

### 使用方法

#### pytorch-retinanet

- 数据准备

```bash
python prepare_data.py
```

- 模型训练

```bash
python train.py --dataset coco --coco_path ./coco --depth 50  --epochs 35 --sample_size 10000 --batch_size 32
```

训练得到：`./logs/train_log.csv`、`./logs/val_logs/csv`、`./logs/model.pt`。

- 验证

```bash
python coco_validation.py --coco_path ./coco --model <your_model_path>.pt
```

#### jittor-retinanet

- 数据准备
- 模型训练
- 验证

### 对齐验证



### 相关资源



| 链接                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| https://drive.google.com/file/d/181qIwc7JePD6m8eJ4O2k7uqVfSiFy4Zg/view | Pytorch + ResNet50 训练模型                                  |
|                                                              | Jittor + ResNet50 训练模型                                   |
| https://www.kaggle.com/datasets/weipengchao/tiny-coco1k      | 抽取自coco2017的小数据集，数据集合结构与coco2017一致，用于快速测试项目功能。 |


