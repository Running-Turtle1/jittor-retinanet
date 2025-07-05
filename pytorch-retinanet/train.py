import argparse
import os
import csv
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import (
    CocoDataset, CSVDataset, collater,
    Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
)
from torch.utils.data import DataLoader
from retinanet import coco_eval, csv_eval

print('CUDA available:', torch.cuda.is_available())

warnings.filterwarnings(
    "ignore",
    message=".*torch.cuda.amp.autocast.*",
    category=FutureWarning
)

def main(args=None):
    parser = argparse.ArgumentParser(description='Training script for RetinaNet with logging.')
    parser.add_argument('--dataset', help='Dataset type: csv or coco.', required=True)
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to CSV training annotations')
    parser.add_argument('--csv_classes', help='Path to CSV class list')
    parser.add_argument('--csv_val', help='Path to CSV validation annotations')
    parser.add_argument('--depth', type=int, default=50, help='ResNet depth: 18, 34, 50, 101, 152')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--sample_size', type=int, default=0,
                        help='Number of samples to randomly select from training dataset; 0 means use full dataset')
    args = parser.parse_args(args)

    # 日志文件准备
    os.makedirs('logs', exist_ok=True)
    train_log_f = open('logs/train_log.csv', 'w', newline='')
    train_logger = csv.writer(train_log_f)
    train_logger.writerow([
        'epoch', 'iter', 'cls_loss', 'reg_loss', 'total_loss',
        'lr', 'time_elapsed', 'img_per_sec'
    ])

    val_log_f = open('logs/val_log.csv', 'w', newline='')
    val_logger = csv.writer(val_log_f)
    val_logger.writerow(['epoch', 'mAP', 'val_time'])

    # 创建数据集和 DataLoader
    if args.dataset == 'coco':
        if not args.coco_path:
            raise ValueError('Must provide --coco_path for COCO dataset')
        dataset_train = CocoDataset(
            args.coco_path, set_name='train2017',
            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        )
        dataset_val = CocoDataset(
            args.coco_path, set_name='val2017',
            transform=transforms.Compose([Normalizer(), Resizer()])
        )
    elif args.dataset == 'csv':
        if not args.csv_train or not args.csv_classes:
            raise ValueError('Must provide --csv_train and --csv_classes for CSV dataset')
        dataset_train = CSVDataset(
            train_file=args.csv_train,
            class_list=args.csv_classes,
            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        )
        if args.csv_val:
            dataset_val = CSVDataset(
                train_file=args.csv_val,
                class_list=args.csv_classes,
                transform=transforms.Compose([Normalizer(), Resizer()])
            )
        else:
            dataset_val = None
            print('No validation annotations provided.')
    else:
        raise ValueError('Dataset type not understood (must be csv or coco)')

    sampler = AspectRatioBasedSampler(
        data_source=dataset_train,
        batch_size=args.batch_size,
        drop_last=False,
        sample_size=args.sample_size
    )
    dataloader_train = DataLoader(
        dataset_train,
        num_workers=3,
        collate_fn=collater,
        batch_sampler=sampler
    )

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(
            data_source=dataset_val,
            batch_size=1,
            drop_last=False,
            sample_size=args.sample_size
        )
        dataloader_val = DataLoader(
            dataset_val,
            num_workers=3,
            collate_fn=collater,
            batch_sampler=sampler_val
        )

    # 构建模型
    if args.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif args.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif args.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif args.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif args.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth')

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True


    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    print('Num training images:', len(dataset_train))
    print('Sample Size:', len(dataset_train) if args.sample_size == 0 else args.sample_size)

    # 训练循环
    for epoch_num in range(args.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        # 本 epoch 的 loss 累积，用于 scheduler
        epoch_losses = []

        for iter_num, data in enumerate(dataloader_train):
            iter_start = time.time()
            optimizer.zero_grad()
            try:
                imgs = data['img'].cuda().float() if torch.cuda.is_available() else data['img'].float()
                cls_loss, reg_loss = retinanet([imgs, data['annot']])
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                total_loss = cls_loss + reg_loss

                if float(total_loss) == 0:
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), max_norm=0.1)
                optimizer.step()

                # 记录并写日志
                lr = optimizer.param_groups[0]['lr']
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

            except Exception as e:
                print('Error during training iteration:', e)
                continue

        # 验证
        if dataset_val is not None:
            val_start = time.time()
            if args.dataset == 'coco':
                mAP = coco_eval.evaluate_coco(dataset_val, retinanet)
            else:
                mAP = csv_eval.evaluate(dataset_val, retinanet)
            val_time = time.time() - val_start
            val_logger.writerow([epoch_num, mAP, round(val_time, 4)])
            print(f"Val Epoch {epoch_num} | mAP {mAP:.3f} | time {val_time:.2f}s")

        # 更新学习率
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            scheduler.step(avg_loss)

        # 保存训练过程中的权重
        torch.save(retinanet.module, f'logs/{args.dataset}_retinanet_epoch{epoch_num}.pt')

    # 训练结束后保存最终模型
    retinanet.eval()
    torch.save(retinanet, 'logs/model_final.pt')

    train_log_f.close()
    val_log_f.close()


if __name__ == '__main__':
    main()
