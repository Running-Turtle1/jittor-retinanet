# from __future__ import print_function, division
import sys
import os
import numpy as np
import random
import csv
import math
import skimage.io
import skimage.transform
import skimage.color
import skimage
import jittor as jt
from pycocotools.coco import COCO
from jittor.dataset.dataset import Dataset


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', batch_size=1, shuffle=False, transform=None):
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
        self.set_attrs(batch_size=self.batch_size, total_len=self.total_len, shuffle=self.shuffle, drop_last=self.drop_last)

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
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
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)
        for a in coco_annotations:
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80

    def collate_batch(self, data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        scales = [s['scale'] for s in data]
        widths = [int(s.shape[0]) for s in imgs]
        heights = [int(s.shape[1]) for s in imgs]
        batch_size = len(imgs)
        max_width = np.array(widths).max()
        max_height = np.array(heights).max()

        padded_imgs = jt.zeros((int(batch_size), int(max_width), int(max_height), 3), dtype='float32')
        for i in range(batch_size):
            img = imgs[i]
            padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

        max_num_annots = max(annot.shape[0] for annot in annots)
        if max_num_annots > 0:
            annot_padded = jt.ones((len(annots), max_num_annots, 5), dtype='float32') * -1
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = jt.ones(shape=(len(annots), 1, 5), dtype='float32') * -1

        padded_imgs = padded_imgs.permute(0, 3, 1, 2)

        return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        try:
            return function(value)
        except ValueError as e:
            raise ValueError(fmt.format(e))

    def _open_for_csv(self, path):
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1
            try:
                class_name, class_id = row
            except ValueError:
                raise ValueError("line {}: format should be 'class_name,class_id'".format(line))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError("line {}: duplicate class name: '{}'".format(line, class_name))
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
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        if len(annotation_list) == 0:
            return annotations

        for a in annotation_list:
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
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise ValueError(
                    "line {}: format should be 'img_file,x1,y1,x2,y2,class_name' or 'img_file,,,,,'".format(line)
                )

            if img_file not in result:
                result[img_file] = []

            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            if class_name not in classes:
                raise ValueError("line {}: unknown class name: '{}' (classes: {})".format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = skimage.io.imread(self.image_names[image_index])
        return float(image.shape[1]) / float(image.shape[0])


class Resizer(object):
    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side

        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        annots[:, :4] *= scale

        return {'img': jt.array(new_image), 'annot': jt.array(annots), 'scale': scale}


class Augmenter(object):
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]
            _, cols, _ = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x1

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class UnNormalizer:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = jt.array(mean).reshape(-1, 1, 1)
        self.std = jt.array(std).reshape(-1, 1, 1)

    def __call__(self, tensor):
        return tensor * self.std + self.mean


class AspectRatioBasedSampler():
    def __init__(self, data_source, batch_size, drop_last=False, sample_size=0):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        total_size = len(data_source)
        if sample_size and sample_size < total_size:
            self.indices = random.sample(range(total_size), sample_size)
        else:
            self.indices = list(range(total_size))

        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return math.ceil(len(self.indices) / self.batch_size)

    def group_images(self):
        order = sorted(self.indices, key=lambda idx: self.data_source.image_aspect_ratio(idx))
        groups = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]

        if groups and not self.drop_last and len(groups[-1]) < self.batch_size:
            last = groups[-1]
            need = self.batch_size - len(last)
            last += order[:need]
            groups[-1] = last

        return groups
