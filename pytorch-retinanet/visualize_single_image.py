import torch
import numpy as np
import time
import os
import csv
import argparse
import cv2
from retinanet import model as retinanet_model  # 使用正确的模型引用


def load_classes(csv_reader):
    result = {}
    next(csv_reader)  # 跳过表头
    for line, row in enumerate(csv_reader):
        line += 1
        try:
            class_name, class_id = row
        except ValueError:
            raise ValueError(f'line {line}: format should be \'class_name,class_id\'')
        class_id = int(class_id)

        if class_name in result:
            raise ValueError(f'line {line}: duplicate class name: \'{class_name}\'')

        result[class_name] = class_id
    return result


# 在图片框上绘制标签
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


# 检测图像并绘制边框
def detect_image(image_path, model_path, class_list, output_dir):
    # 读取类名和ID
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter = ','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    # 定义模型架构（假设是 RetinaNet）
    model = retinanet_model.resnet50(num_classes = len(classes), pretrained = True)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path))

    # 如果有GPU, 使用GPU
    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    # 遍历图像并进行检测
    for img_name in os.listdir(image_path):
        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # 缩放图像
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # 调整图像大小
        image = cv2.resize(image, (int(round(cols * scale)), int(round(rows * scale))))
        rows, cols, cns = image.shape

        # 填充图像的宽高为32的倍数
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():
            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            # 推理
            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print(f'Elapsed time: {time.time() - st}')

            # 获取得分大于 0.5 的索引
            idxs = np.where(scores.cpu() > 0.5)

            # 绘制检测框和标签
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                score = scores[j]
                caption = f'{label_name} {score:.3f}'

                # 绘制边框和标签
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color = (0, 0, 255), thickness = 2)

            # 保存图像到输出目录
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, image_orig)
            print(f"Saved visualized image to {output_path}")


if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser(description = 'Simple script for visualizing result of training.')

    # 添加命令行参数
    parser.add_argument('--image_dir', help = 'Path to directory containing images')
    parser.add_argument('--model_path', help = 'Path to model')
    parser.add_argument('--class_list', help = 'Path to CSV file listing class names (see README)')
    parser.add_argument('--output_dir', help = 'Directory to save the visualized images')

    # 解析命令行参数
    args = parser.parse_args()

    # 创建输出目录（如果不存在）
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 调用图像检测函数
    detect_image(args.image_dir, args.model_path, args.class_list, args.output_dir)
