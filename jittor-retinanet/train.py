import argparse
import csv
import os
import subprocess
import time

import jittor as jt
import numpy as np
from jittor import optim, transform
from jittor.lr_scheduler import ReduceLROnPlateau

from retinanet import coco_eval, model
from retinanet.dataloader import CocoDataset, Augmenter, Normalizer, Resizer


print('CUDA available:', jt.has_cuda)


def build_model(depth, num_classes):
    if depth != 50:
        raise ValueError('Jittor implementation currently supports only --depth 50')
    return model.resnet50(num_classes=num_classes, pretrained=True)


def get_gpu_mem_mb():
    if not jt.has_cuda:
        return -1.0

    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True,
        )
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        logical_idx = 0
        physical_idx = logical_idx
        if visible:
            visible_ids = [item.strip() for item in visible.split(',') if item.strip()]
            if logical_idx < len(visible_ids) and visible_ids[logical_idx].isdigit():
                physical_idx = int(visible_ids[logical_idx])
        values = [line.strip() for line in output.splitlines() if line.strip()]
        if physical_idx < len(values):
            return float(values[physical_idx])
    except Exception:
        pass
    return -1.0


def main(args=None):
    parser = argparse.ArgumentParser(description='Training script for RetinaNet with logging.')
    parser.add_argument('--coco_path', default='./coco', help='Path to COCO directory')
    parser.add_argument('--depth', type=int, default=50, help='ResNet depth: currently only 50 is supported')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')

    args = parser.parse_args(args)

    if not args.coco_path:
        raise ValueError('Must provide --coco_path for COCO dataset')

    if jt.has_cuda:
        jt.flags.use_cuda = 1

    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    train_log_f = open('logs/train_log.csv', 'w', newline='')
    train_logger = csv.writer(train_log_f)
    train_logger.writerow([
        'epoch', 'iter', 'global_step', 'cls_loss', 'reg_loss', 'total_loss',
        'lr', 'time_elapsed', 'img_per_sec', 'gpu_mem_mb'
    ])

    val_log_f = open('logs/val_log.csv', 'w', newline='')
    val_logger = csv.writer(val_log_f)
    val_logger.writerow([
        'epoch', 'global_step', 'mAP', 'AP50', 'AP75',
        'val_time', 'epoch_time_sec', 'avg_img_per_sec', 'gpu_mem_mb'
    ])

    dataloader_train = CocoDataset(
        root_dir=args.coco_path,
        set_name='train2017',
        batch_size=args.batch_size,
        shuffle=True,
        transform=transform.Compose([Normalizer(), Augmenter(), Resizer()])
    )
    dataset_val = CocoDataset(
        root_dir=args.coco_path,
        set_name='val2017',
        batch_size=1,
        shuffle=False,
        transform=transform.Compose([Normalizer(), Resizer()])
    )

    retinanet = build_model(args.depth, dataloader_train.num_classes())
    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    print('Num training images:', dataloader_train.total_len)
    global_step = 0

    for epoch_num in range(args.epochs):
        epoch_start = time.time()
        retinanet.train()
        retinanet.freeze_bn()
        epoch_losses = []

        for iter_num, data in enumerate(dataloader_train):
            iter_start = time.time()

            cls_loss, reg_loss = retinanet([data['img'], data['annot']])
            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            total_loss = cls_loss + reg_loss

            if float(total_loss) == 0:
                continue

            optimizer.step(total_loss)
            global_step += 1

            lr = 1e-5
            iter_time = time.time() - iter_start
            img_per_sec = data['img'].shape[0] / iter_time
            gpu_mem_mb = get_gpu_mem_mb()
            train_logger.writerow([
                epoch_num, iter_num, global_step,
                float(cls_loss), float(reg_loss), float(total_loss),
                lr, round(iter_time, 4), round(img_per_sec, 2), gpu_mem_mb
            ])
            epoch_losses.append(float(total_loss))

            print(
                f"Epoch {epoch_num} | Iter {iter_num} |"
                f" cls_loss {cls_loss:.4f} | reg_loss {reg_loss:.4f} |"
                f" total_loss {total_loss:.4f} | lr {lr:.1e} |"
                f" {img_per_sec:.1f} img/s"
            )

        val_start = time.time()
        metrics = coco_eval.evaluate_coco(dataset_val, retinanet, results_dir='results')
        val_time = time.time() - val_start
        epoch_time_sec = time.time() - epoch_start
        avg_img_per_sec = dataloader_train.total_len / epoch_time_sec if epoch_time_sec > 0 else 0.0
        gpu_mem_mb = get_gpu_mem_mb()
        mAP = metrics['map'] if metrics is not None else -1.0
        ap50 = metrics['ap50'] if metrics is not None else -1.0
        ap75 = metrics['ap75'] if metrics is not None else -1.0
        val_logger.writerow([
            epoch_num, global_step, mAP, ap50, ap75,
            round(val_time, 4), round(epoch_time_sec, 4), round(avg_img_per_sec, 2), gpu_mem_mb
        ])
        print(f"Val Epoch {epoch_num} | mAP {mAP:.3f} | AP50 {ap50:.3f} | AP75 {ap75:.3f} | time {val_time:.2f}s")

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            scheduler.step(avg_loss)

        jt.save(retinanet.state_dict(), f'checkpoints/coco_retinanet_epoch{epoch_num}.pt')

    retinanet.eval()
    jt.save(retinanet.state_dict(), 'checkpoints/model_final.pt')

    train_log_f.close()
    val_log_f.close()


if __name__ == '__main__':
    main()
