from __future__ import print_function, division
import sys
import os
import numpy as np
import random
import csv
import math
import jittor as jt
from pycocotools.coco import COCO
from PIL import Image
import skimage.io
import skimage.transform
import skimage.color
import skimage
from jittor.dataset.dataset import Dataset

class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name = 'train2017', transform = None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()

        self.root_dir = root_dir  # COCO 数据集的根目录
        self.set_name = set_name  # 'train2017' 或 'val2017' 等，指定数据集的子集
        self.transform = transform  # 数据预处理

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.set_attrs(
            total_len = len(self.image_ids),
            # sampler = AspectRatioBasedSampler,
        )

        # 设置Jittor数据集的基本属性
        # self.total_len = len(self.image_ids)

        self.collate_fn = collate_batch
        self.batch_sampler = AspectRatioBasedSampler
        self.sample_size = 0

        self.load_classes()

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
        # [{'supercategory': 'person', 'id': 1, 'name': 'person'},
        #  {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
        #  {'supercategory': 'vehicle', 'id': 3, 'name': 'bus'}]

        # self.classes: 类别名 → 连续的训练标签
        # self.classes = {
        #     'person': 0,
        #     'bicycle': 1,
        #     ...
        # }
        self.classes = {}
        # self.coco_labels: 训练标签 → 原始COCO类别ID
        # self.coco_labels = {
        #     0: 1,  # 训练标签0 是 COCO 类别 ID 1
        #     1: 2,
        #     ...
        # }
        self.coco_labels = {}
        # self.coco_labels_inverse: COCO类别ID → 训练标签
        # self.coco_labels_inverse = {
        #     1: 0,
        #     2: 1,
        #     ...
        # }
        self.coco_labels_inverse = {}

        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
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
        # loadImages:
        # [{'license': 3,
        #   'file_name': '000000391895.jpg',
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000391895.jpg',
        #   'height': 360,
        #   'width': 640,
        #   'date_captured': '2013-11-14 11:18:45',
        #   'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
        #   'id': 391895}]
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        # img 返回的图像, 是一个 Numpy 数组, 形状通常是
        # (H, W, 3) : RGB 彩色图像（高、宽、3 通道）
        # (H, W) : 灰色图像（无颜色通道）
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
        # get ground truth annotations
        # 参数 iscrowd=False 是为了排除“crowd”类型的标注（难以单独检测的群体标注）
        # E.G. [151091, 202758, 1260346, 1766676]
        annotations_ids = self.coco.getAnnIds(imgIds = self.image_ids[image_index], iscrowd = False)
        # 初始化一个空的 NumPy 数组，用于存放所有目标的标注信息，5列分别是 [x1, y1, x2, y2, label]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        # E.G.
        # {'segmentation': [[376.97,
        #                    ..........
        #                    184.19]],
        #  'area': 12190.44565,
        #  'iscrowd': 0,
        #  'image_id': 391895,
        #  'bbox': [359.17, 146.17, 112.45, 213.57],
        #  'category_id': 4,
        #  'id': 151091}
        coco_annotations = self.coco.loadAnns(annotations_ids)
        # a['bbox']: [x, y, w, h]
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


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform = None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter = ','))
        except ValueError as e:
            raise (ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter = ','), self.classes)
        except ValueError as e:
            raise (ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise ValueError(fmt.format(e))

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline = '')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a['class'])
            annotations = np.append(annotations, annotation, axis = 0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise ValueError(
                    'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line))

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    """把这一批（batch）里尺寸和框数量都不一样的样本，统一成同样大小的张量，才能一次性喂给模型"""
    imgs = [s['img'] for s in data] # (B, W, H, C) -> B * (W, H, C)
    annots = [s['annot'] for s in data] # (B, N, R^5) -> B * (N, R^5)
    scales = [s['scale'] for s in data] # (B, float) -> B * (float)
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
        annot_padded = jt.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        # 如果所有样本都没有框（max_num_annots == 0），则直接用 (batch_size,1,5) 的 −1 张量代替
        annot_padded = jt.ones((len(annots), 1, 5)) * -1

    #  (batch, W, H, C) → (batch, C, W, H)
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

def collate_batch(self, batch):
    return collater(batch)

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

    def __call__(self, sample, flip_x=0.5):
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


class UnNormalizer(object):
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


# class AspectRatioBasedSampler(Sampler):
#     """
#     按照图像的长宽比分组采样，目的是让一个 batch 中图像尺寸更接近，
#     从而提升训练效率并减少 padding 的浪费。
#     drop_last: 是否丢弃最后不足一个 batch 的部分
#     groups: 存储按 aspect ratio 排序后的分组列表
#     """
#     def __init__(self, data_source, batch_size, drop_last):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.groups = self.group_images()
#
#     def __iter__(self):
#         random.shuffle(self.groups) # 打乱以增加数据多样性
#         for group in self.groups:
#             yield group
#
#     def __len__(self):
#         if self.drop_last:
#             return len(self.data_source) // self.batch_size
#         else:
#             return (len(self.data_source) + self.batch_size - 1) // self.batch_size  # ceil(x, y)
#
#     def group_images(self):
#         # determine the order of the images by aspect ratio
#         order = list(range(len(self.data_source)))
#         order.sort(key = lambda x: self.data_source.image_aspect_ratio(x))
#
#         # divide into groups, one group = one batch
#         # The reason for using x % len(order) is to enable cyclic sampling when drop_last is set to False
#         return [
#             [order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)
#         ]
# class AspectRatioBasedSampler(Sampler):
class AspectRatioBasedSampler():
    """
    按照图像的长宽比分组采样，同时支持从全量数据中随机抽取子集再做分组。

    参数:
        data_source: 数据集对象，需实现 image_aspect_ratio(idx) 方法
        batch_size:  每个 batch 的样本数
        drop_last:  是否丢弃最后一个不满 batch 的分组
        sample_size: 如果 >0，则先从全量数据中随机抽取 sample_size 个样本
    """
    def __init__(self, data_source, batch_size, drop_last=False, sample_size=0):
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
        order = sorted(self.indices, key=lambda idx: self.data_source.image_aspect_ratio(idx))

        # 再按 batch_size 划分成若干组
        groups = [
            order[i : i + self.batch_size]
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

class BatchDataLoader:
    def __init__(self, dataset, sampler, collate_fn):
        self.dataset = dataset
        self.sampler = sampler
        self.collate_fn = collate_fn

    def __iter__(self):
        for indices in self.sampler:
            samples = [self.dataset[idx] for idx in indices]
            yield self.collate_fn(samples)

    def __len__(self):
        return len(self.sampler)