import argparse
import os
import csv
import time
# 移除了针对 PyTorch 的 warnings 导入，因为已不再需要

import numpy as np
from jittor.lr_scheduler import ReduceLROnPlateau
from jittor.dataset.dataset import Dataset
from myutils.optim import clip_grad_norm_
from myretinanet.dataloader import (
    CocoDataset, Augmenter, Normalizer, Resizer
)
from retinanet import model
from retinanet import coco_eval
import jittor as jt
from jittor import transform, optim


print('CUDA available:', jt.has_cuda)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args = None):
    parser = argparse.ArgumentParser(description = 'Training script for RetinaNet with logging.')
    parser.add_argument('--dataset', default = 'tiny_coco', help = 'Dataset type: csv or coco.')  # 默认值改为 'coco'
    parser.add_argument('--coco_path', default = './tiny_coco', help = 'Path to COCO directory')

    parser.add_argument('--depth', type = int, default = 50, help = 'ResNet depth: 18, 34, 50, 101, 152')
    parser.add_argument('--epochs', type = int, default = 5, help = 'Number of epochs')
    parser.add_argument('--batch_size', type = int, default = 8, help = 'Batch size')
    parser.add_argument('--sample_size', type = int, default = 100,
                        help = 'Number of samples to randomly select from training dataset; 0 means use full dataset')
    args = parser.parse_args(args)

    if jt.has_cuda:
        jt.flags.use_cuda = 1

    # 日志文件准备
    os.makedirs('logs', exist_ok = True)
    train_log_f = open('logs/train_log.csv', 'w', newline = '')
    train_logger = csv.writer(train_log_f)
    train_logger.writerow([
        'epoch', 'iter', 'cls_loss', 'reg_loss', 'total_loss',
        'lr', 'time_elapsed', 'img_per_sec'
    ])

    val_log_f = open('logs/val_log.csv', 'w', newline = '')
    val_logger = csv.writer(val_log_f)
    val_logger.writerow(['epoch', 'mAP', 'val_time'])

    dataloader_train = CocoDataset(
        root_dir = './coco',
        set_name = 'train2017',
        batch_size = 2,
        shuffle = True,
        transform = transform.Compose([Normalizer(), Resizer()])
    )

    dataloader_val = CocoDataset(
        root_dir = './tiny_coco',
        set_name = 'val2017',
        batch_size = 1,
        shuffle = True,
        transform = transform.Compose([Normalizer(), Resizer()])
    )

    # 构建模型
    retinanet = model.resnet50(num_classes = dataloader_train.num_classes(), pretrained = True)
    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr = 1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience = 3, verbose = True)

    print('Num training images:', dataloader_train.total_len)
    # print(optimizer.param_groups)
    # return
    # 训练循环
    for epoch_num in range(args.epochs):
        print(f'Start {epoch_num} epoch!')
        # print('before train')
        retinanet.train()
        # print('after train')
        retinanet.freeze_bn()

        # 本 epoch 的 loss 累积，用于 scheduler
        epoch_losses = []



        for iter_num, data in enumerate(dataloader_train):


            iter_start = time.time()
            cls_loss, reg_loss = retinanet([data['img'], data['annot']])
            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            total_loss = cls_loss + reg_loss

            optimizer.step(total_loss)


            lr = 1e-5 # optimizer.param_groups[0]['lr']
            iter_time = time.time() - iter_start
            img_per_sec = data['img'].shape[0] / iter_time
            train_logger.writerow([
                epoch_num, iter_num,
                float(cls_loss), float(reg_loss), float(total_loss),
                lr, round(iter_time, 4), round(img_per_sec, 2)
            ])
            epoch_losses.append(float(total_loss))

            print(f"Epoch {epoch_num} | Iter {iter_num} |"
                  f" cls_loss {cls_loss:.4f} | reg_loss {reg_loss:.4f} |"
                  f" total_loss {total_loss:.4f} | lr {lr:.1e} |"
                  f" {img_per_sec:.1f} img/s")

        # 验证
        # val_start = time.time()
        # # 由于移除了 CSV 参数，这里假定只处理 COCO
        # mAP = coco_eval.evaluate_coco(dataloader_val, retinanet)
        # val_time = time.time() - val_start
        # val_logger.writerow([epoch_num, mAP, round(val_time, 4)])
        # print(f"Val Epoch {epoch_num} | mAP {mAP:.3f} | time {val_time:.2f}s")

        # 更新学习率
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            scheduler.step(avg_loss)

        # 保存训练过程中的权重
        # 修正：保存模型的 state_dict，更通用且避免 retinanet.module 的歧义
        jt.save(retinanet.state_dict(), f'logs/{args.dataset}_retinanet_epoch{epoch_num}.pt')

        # 训练结束后保存最终模型
    retinanet.eval()
    jt.save(retinanet.state_dict(), 'logs/model_final.pt')  # 修正：保持与训练中保存方式一致

    train_log_f.close()
    val_log_f.close()


if __name__ == '__main__':

    main()