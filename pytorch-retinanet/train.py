import argparse
import csv
import os
import subprocess
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet import coco_eval, model
from retinanet.dataloader import (
    CocoDataset, collater,
    Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
)

print('CUDA available:', torch.cuda.is_available())


def get_gpu_mem_mb():
    if not torch.cuda.is_available():
        return -1.0

    try:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        logical_idx = torch.cuda.current_device()
        physical_idx = logical_idx
        if visible:
            visible_ids = [item.strip() for item in visible.split(',') if item.strip()]
            if logical_idx < len(visible_ids) and visible_ids[logical_idx].isdigit():
                physical_idx = int(visible_ids[logical_idx])
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True,
        )
        values = [line.strip() for line in output.splitlines() if line.strip()]
        if physical_idx < len(values):
            return float(values[physical_idx])
    except Exception:
        pass
    return -1.0



def main(args = None):
    parser = argparse.ArgumentParser(description = 'Training script for RetinaNet with logging.')
    parser.add_argument('--coco_path', default = './coco', help = 'Path to COCO directory')
    parser.add_argument('--depth', type = int, default = 50, help = 'ResNet depth: 18, 34, 50, 101, 152')
    parser.add_argument('--epochs', type = int, default = 5, help = 'Number of epochs')
    parser.add_argument('--batch_size', type = int, default = 2, help = 'Batch size')

    args = parser.parse_args(args)

    # 日志文件准备
    os.makedirs('logs', exist_ok = True)
    os.makedirs('checkpoints', exist_ok = True)
    os.makedirs('results', exist_ok = True)
    train_log_f = open('logs/train_log.csv', 'w', newline = '')
    train_logger = csv.writer(train_log_f)
    train_logger.writerow([
        'epoch', 'iter', 'global_step', 'cls_loss', 'reg_loss', 'total_loss',
        'lr', 'time_elapsed', 'img_per_sec', 'gpu_mem_mb'
    ])

    val_log_f = open('logs/val_log.csv', 'w', newline = '')
    val_logger = csv.writer(val_log_f)
    val_logger.writerow([
        'epoch', 'global_step', 'mAP', 'AP50', 'AP75',
        'val_time', 'epoch_time_sec', 'avg_img_per_sec', 'gpu_mem_mb'
    ])

    # 创建数据集和 DataLoader
    if not args.coco_path:
        raise ValueError('Must provide --coco_path for COCO dataset')
    dataset_train = CocoDataset(
        args.coco_path, set_name = 'train2017',
        transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
    )
    dataset_val = CocoDataset(
        args.coco_path, set_name = 'val2017',
        transform = transforms.Compose([Normalizer(), Resizer()])
    )

    sampler = AspectRatioBasedSampler(
        data_source = dataset_train,
        batch_size = args.batch_size,
        drop_last = False
    )
    dataloader_train = DataLoader(
        dataset_train,
        num_workers = 3,
        collate_fn = collater,
        batch_sampler = sampler
    )

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(
            data_source = dataset_val,
            batch_size = 1,
            drop_last = False
        )
        dataloader_val = DataLoader(
            dataset_val,
            num_workers = 3,
            collate_fn = collater,
            batch_sampler = sampler_val
        )

    # 构建模型
    if args.depth == 18:
        retinanet = model.resnet18(num_classes = dataset_train.num_classes(), pretrained = True)
    elif args.depth == 34:
        retinanet = model.resnet34(num_classes = dataset_train.num_classes(), pretrained = True)
    elif args.depth == 50:
        retinanet = model.resnet50(num_classes = dataset_train.num_classes(), pretrained = True)
    elif args.depth == 101:
        retinanet = model.resnet101(num_classes = dataset_train.num_classes(), pretrained = True)
    elif args.depth == 152:
        retinanet = model.resnet152(num_classes = dataset_train.num_classes(), pretrained = True)
    else:
        raise ValueError('Unsupported model depth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retinanet = retinanet.to(device)


    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr = 1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 3, verbose = True)

    print('Num training images:', len(dataset_train))
    global_step = 0

    # 训练循环
    for epoch_num in range(args.epochs):
        epoch_start = time.time()
        retinanet.train()
        retinanet.freeze_bn()
        epoch_losses = []

        for iter_num, data in enumerate(dataloader_train):
            iter_start = time.time()
            optimizer.zero_grad()
            try:
                # imgs = data['img'].cuda().float() if torch.cuda.is_available() else data['img'].float()
                # cls_loss, reg_loss = retinanet([imgs, data['annot']])
                imgs = data['img'].to(device).float()
                annots = data['annot'].to(device)
                cls_loss, reg_loss = retinanet([imgs, annots])
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                total_loss = cls_loss + reg_loss

                if float(total_loss) == 0:
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), max_norm = 0.1)
                optimizer.step()
                global_step += 1

                # 记录并写日志
                lr = optimizer.param_groups[0]['lr']
                iter_time = time.time() - iter_start
                img_per_sec = data['img'].shape[0] / iter_time
                gpu_mem_mb = get_gpu_mem_mb()
                train_logger.writerow([
                    epoch_num, iter_num, global_step,
                    float(cls_loss), float(reg_loss), float(total_loss),
                    lr, round(iter_time, 4), round(img_per_sec, 2), gpu_mem_mb
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
            metrics = coco_eval.evaluate_coco(dataset_val, retinanet.to(device), results_dir = 'results')
            val_time = time.time() - val_start
            epoch_time_sec = time.time() - epoch_start
            avg_img_per_sec = len(dataset_train) / epoch_time_sec if epoch_time_sec > 0 else 0.0
            gpu_mem_mb = get_gpu_mem_mb()
            mAP = metrics['map'] if metrics is not None else -1.0
            ap50 = metrics['ap50'] if metrics is not None else -1.0
            ap75 = metrics['ap75'] if metrics is not None else -1.0
            val_logger.writerow([
                epoch_num, global_step, mAP, ap50, ap75,
                round(val_time, 4), round(epoch_time_sec, 4), round(avg_img_per_sec, 2), gpu_mem_mb
            ])
            print(f"Val Epoch {epoch_num} | mAP {mAP:.3f} | AP50 {ap50:.3f} | AP75 {ap75:.3f} | time {val_time:.2f}s")

        # 更新学习率
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            scheduler.step(avg_loss)

        # 保存训练过程中的权重
        torch.save(retinanet, f'checkpoints/coco_retinanet_epoch{epoch_num}.pt')

    # 训练结束后保存最终模型
    retinanet.eval()
    torch.save(retinanet, 'checkpoints/model_final.pt')

    train_log_f.close()
    val_log_f.close()


if __name__ == '__main__':
    main()
