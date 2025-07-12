# from __future__ import print_function, division
import sys
import os
import numpy as np
import random
import csv
import math
# from torch.utils.data.sampler import Sampler
import skimage.io
import skimage.transform
import skimage.color
import skimage
import jittor as jt
from pycocotools.coco import COCO
from jittor.dataset.dataset import Dataset

class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name = 'train2017', batch_size = 1, shuffle = False, transform = None):
        super(CocoDataset, self).__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.drop_last = True

        self.load_classes()

        self.total_len = len(self.image_ids)
        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle, drop_last = self.drop_last)

    def load_classes(self):
        """
        核心工作就是：把 COCO 原始类别 ID（稀疏、不连续）映射成从 0 开始的密集连续标签。
        self.classes 类别名 → 连续的训练标签 person:1
        self.coco_labels 连续的训练标签 → 类别ID 0:1
        self.coco_labels_inverse 类别ID → 连续的训练标签 1:0
        """
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key = lambda x: x['id'])
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}

        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        """
        返回 COCO 数据集中用于训练的图像数量
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        定义用 dataset[idx] 这种方式访问数据时的返回对象 {img, annot} (optional: transform)
        """
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        """
        给定一个图片索引 image_index，返回对应图片的归一化图像数据（float32 格式，范围 0~1）。
        """
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        # 将灰度图复制三份作为 RGB 通道，变成彩色格式
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        """
        根据给定的图片索引 image_index，从 COCO 数据集中获取该图片的所有目标标注信息，处理后返回一个数组，格式是：
        [[x1, y1, x2, y2, label],
         ...
         ]
        """
        annotations_ids = self.coco.getAnnIds(imgIds = self.image_ids[image_index], iscrowd = False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis = 0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        """ coco'label --> train label """
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        """ train label --> coco'label """
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        """获取 image_index 图像对应的 aspect_ratio"""
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        """coco固定类别数"""
        return 80


    def collate_batch(self, data):
        """把这一批（batch）里尺寸和框数量都不一样的样本，统一成同样大小的张量，才能一次性喂给模型"""
        imgs = [s['img'] for s in data]  # (B, W, H, C) -> B * (W, H, C)
        annots = [s['annot'] for s in data]  # (B, N, R^5) -> B * (N, R^5)
        scales = [s['scale'] for s in data]  # (B, float) -> B * (float)
        widths = [int(s.shape[0]) for s in imgs]
        heights = [int(s.shape[1]) for s in imgs]
        batch_size = len(imgs)
        max_width = np.array(widths).max()
        max_height = np.array(heights).max()

        # 填充
        padded_imgs = jt.zeros((int(batch_size), int(max_width), int(max_height), 3), dtype = "float32")
        for i in range(batch_size):
            img = imgs[i]
            # 把原图拷贝到左上角，其它区域保持 0
            padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

        max_num_annots = max(annot.shape[0] for annot in annots)
        if max_num_annots > 0:
            annot_padded = jt.ones((len(annots), max_num_annots, 5), dtype = "float32") * -1
            if max_num_annots > 0:
                for idx, annot in enumerate(annots):
                    # print(annot.shape)
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = annot
        else:
            # 如果所有样本都没有框（max_num_annots == 0），则直接用 (batch_size,1,5) 的 −1 张量代替
            annot_padded = jt.ones(shape = (len(annots), 1, 5), dtype = "float32") * -1

        #  (batch, W, H, C) → (batch, C, W, H)
        padded_imgs = padded_imgs.permute(0, 3, 1, 2)

        return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """对图像进行缩放和填充，以便输入到 RetinaNet 使用, 并且会同步调整目标检测框（annots）的坐标"""

    def __call__(self, sample, min_side = 608, max_side = 1024):
        # E.G.1 sample['img'].shape -> (360, 640, 3)
        # E.G.2 sample['annot'].shape -> (4, 5) per element: [x1,x2,y1,y2,index]
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # 默认希望图像的最短边缩放到 min_side（如608）。
        scale = min_side / smallest_side

        # 若缩放后最长边超过 max_side（如1024），就重新调整 scale，以避免过大图像。
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        # 填充图像尺寸为 32 的倍数
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        # 同步调整标注框（annots）
        annots[:, :4] *= scale

        # 返回字典格式的结果
        return {'img': jt.array(new_image), 'annot': jt.array(annots), 'scale': scale}


class Augmenter(object):
    """水平翻转图像并同步调整标注框坐标。"""

    def __call__(self, sample, flip_x = 0.5):
        # 50% 概率进行水平翻转
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            # 图像左右翻转
            image = image[:, ::-1, :]
            _, cols, _ = image.shape

            # 缓存原始 x1, x2
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            # 翻转坐标：new_x1 = cols - old_x2, new_x2 = cols - old_x1
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x1

            # 更新 sample
            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):
    """对图像进行归一化处理"""

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}



class UnNormalizer:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = jt.array(mean).reshape(-1, 1, 1)  # shape [3,1,1]
        self.std = jt.array(std).reshape(-1, 1, 1)    # shape [3,1,1]

    def __call__(self, tensor):
        """
        tensor: jt.Var, shape [3, H, W], normalized image tensor
        returns: jt.Var, unnormalized tensor
        """
        return tensor * self.std + self.mean

# Sampler
class AspectRatioBasedSampler():
    """
    按照图像的长宽比分组采样，同时支持从全量数据中随机抽取子集再做分组。

    参数:
        data_source: 数据集对象，需实现 image_aspect_ratio(idx) 方法
        batch_size:  每个 batch 的样本数
        drop_last:  是否丢弃最后一个不满 batch 的分组
        sample_size: 如果 >0，则先从全量数据中随机抽取 sample_size 个样本
    """

    def __init__(self, data_source, batch_size, drop_last = False, sample_size = 0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 如果指定了 sample_size，则先随机抽样部分索引
        total_size = len(data_source)
        if sample_size and sample_size < total_size:
            self.indices = random.sample(range(total_size), sample_size)
        else:
            self.indices = list(range(total_size))

        # 根据抽样后的 indices 进行 aspect-ratio 分组
        self.groups = self.group_images()

    def __iter__(self):
        # 每个 epoch 先打乱组顺序
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        # 分组后的 batch 数
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return math.ceil(len(self.indices) / self.batch_size)

    def group_images(self):
        # 先按长宽比对采样后的索引排序
        order = sorted(self.indices, key = lambda idx: self.data_source.image_aspect_ratio(idx))

        # 再按 batch_size 划分成若干组
        groups = [
            order[i: i + self.batch_size]
            for i in range(0, len(order), self.batch_size)
        ]

        # 如果不 drop_last，并且最后一组不足 batch_size，可以循环补齐
        if not self.drop_last and len(groups[-1]) < self.batch_size:
            last = groups[-1]
            need = self.batch_size - len(last)
            # 从头部循环取
            last += order[:need]
            groups[-1] = last

        return groups


