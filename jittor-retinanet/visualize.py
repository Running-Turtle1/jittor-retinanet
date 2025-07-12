import numpy as np
import time
import os
import copy
import pdb
import time
import argparse
import sys
import cv2
import jittor as jt
from myretinanet.dataloader import CocoDataset, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from jittor import transform, optim

from retinanet import model


def main(args = None):
    jt.flags.use_cuda = 1

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default = 'tiny_coco', help = 'Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', default = './tiny_coco', help = 'Path to COCO directory')
    parser.add_argument('--model', default = './tiny_coco_retinanet_epoch4.pt', help = 'Path to model file.')
    parser = parser.parse_args(args)

    dataloader_val = CocoDataset(
        root_dir = './tiny_coco',
        set_name = 'val2017',
        batch_size = 1,
        shuffle = True,
        transform = transform.Compose([Normalizer(), Resizer()])
    )

    retinanet = model.resnet50(num_classes = dataloader_val.num_classes(), pretrained = True)
    retinanet.load(parser.model)

    retinanet.eval()

    # use_gpu = True

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):

        with jt.no_grad():
            st = time.time()

            img_input = data['img'].float32()
            scores, classification, transformed_anchors = retinanet(img_input)[0]

            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                # 取出框并转为 numpy
                bbox = transformed_anchors[idxs[0][j]].numpy()
                x1, y1, x2, y2 = map(int, bbox)

                # 获取标签名称
                label_idx = int(classification[idxs[0][j]].numpy())
                label_name = dataloader_val.labels[label_idx]

                # 绘制框和文字
                draw_caption(img, (x1, y1, x2, y2), label_name)
                cv2.rectangle(img, (x1, y1), (x2, y2), color = (0, 0, 255), thickness = 2)
                print(label_name)

            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            os.makedirs('visual_results', exist_ok = True)
            cv2.imwrite(f'visual_results/image_{idx}.jpg', img)


if __name__ == '__main__':
    main()
