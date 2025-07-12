import numpy as np
import jittor as jt
import jittor.nn as nn


def calc_iou(a, b):
    # a: (num_anchors, 4), b: (num_gt, 4)
    # Compute area of ground truths
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    # Intersection widths and heights
    iw = jt.minimum(jt.unsqueeze(a[:, 2], dim = 1), b[:, 2]) - jt.maximum(jt.unsqueeze(a[:, 0], dim = 1), b[:, 0])
    ih = jt.minimum(jt.unsqueeze(a[:, 3], dim = 1), b[:, 3]) - jt.maximum(jt.unsqueeze(a[:, 1], dim = 1), b[:, 1])

    # Clamp to zero
    iw = jt.clamp(iw, min_v = 0)
    ih = jt.clamp(ih, min_v = 0)

    # Union area
    ua = jt.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim = 1) + area - iw * ih
    ua = jt.clamp(ua, min_v = 1e-8)

    # IoU
    inter = iw * ih
    return inter / ua


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        cls_losses = []
        reg_losses = []

        # 假设 anchors[0] 适用于整个批次的锚点计算
        anchor = anchors[0]
        widths = anchor[:, 2] - anchor[:, 0]
        heights = anchor[:, 3] - anchor[:, 1]
        ctr_x = anchor[:, 0] + 0.5 * widths
        ctr_y = anchor[:, 1] + 0.5 * heights

        for i in range(batch_size):
            cls = classifications[i]
            reg = regressions[i]
            bbox_ann = annotations[i]
            # 过滤掉填充用的 -1 标记的真实框
            bbox_ann = bbox_ann[bbox_ann[:, 4] != -1]

            # 钳位分类预测值，避免 log(0) 或 log(1)
            cls = jt.clamp(cls, 1e-4, 1.0 - 1e-4)

            #  print(f"\n--- 批次 {i} 调试信息 ---")
            #  print(f"真实框数量 (过滤后): {bbox_ann.shape[0]}")

            if bbox_ann.shape[0] == 0:  # 如果当前图像没有真实框
                #  print("当前图像没有真实框，只计算负样本分类损失。")
                alpha_factor = jt.ones(cls.shape) * alpha
                alpha_factor = 1. - alpha_factor
                focal_weight = cls
                focal_weight = alpha_factor * jt.pow(focal_weight, gamma)
                bce = -jt.log(1.0 - cls)  # 只有负样本的 BCE
                loss_cls = (focal_weight * bce).sum()
                cls_losses.append(loss_cls)
                reg_losses.append(jt.zeros((1,)))  # 回归损失为 0
                #  print(f"批次 {i} 分类损失: {loss_cls.item():.6f}, 回归损失: 0.0")
                continue

            ious = calc_iou(anchor, bbox_ann[:, :4])
            #  print(f"IoUs 形状: {ious.shape}")
            # 打印部分 IoU 值以便检查，如果锚点很多，只打印前几个
            #  print(f"前5个锚点与所有真实框的IoU:\n{ious[:5, :].numpy()}")

            iou_argmax, iou_max = jt.arg_reduce(ious, 'max', dim = 1, keepdims = False)
            #  print(f"IoU_max (前10个): {iou_max[:10].numpy()}")
            #  print(f"IoU_argmax (前10个): {iou_argmax[:10].numpy()}")

            targets = jt.ones(cls.shape) * -1  # 初始化目标标签为 -1 (忽略)
            #  print(f"Targets 初始化 (前10个，第一列): {targets[:10, 0].numpy()}")

            # 将 IoU 小于 0.4 的锚点标记为负样本 (目标为 0)
            neg_count_before = (targets == 0).sum().item()
            targets[iou_max < 0.4] = 0
            neg_count_after = (targets == 0).sum().item()
            #  print(f"IoU < 0.4 的锚点数量: {neg_count_after - neg_count_before}")
            #  print(f"Targets 负样本赋值后 (前10个，第一列): {targets[:10, 0].numpy()}")

            pos_inds = iou_max >= 0.5  # 找到 IoU 大于等于 0.5 的正样本锚点
            num_pos = pos_inds.sum()  # 统计正样本锚点数量
            #  print(f"正样本锚点数量 (num_pos): {num_pos.item()}")

            assigned = bbox_ann[iou_argmax]  # 获取为每个锚点分配的真实框

            # 为正样本锚点设置目标标签
            targets[pos_inds, :] = 0  # 先将正样本锚点对应的所有类别预测都设为0
            # 然后将其对应分配的真实框的类别设为1
            targets[pos_inds, assigned[pos_inds, 4].long()] = 1
            #  print(f"Targets 正样本赋值后 (前10个，第一列): {targets[:10, 0].numpy()}")
            #  print(f"Targets 赋值后 (前10个, 所有列): \n{targets[:10].numpy()}")  # 打印更多列以检查类别分配

            # Focal Loss 权重计算
            alpha_factor = jt.where(targets == 1, jt.ones_like(targets) * alpha, jt.ones_like(targets) * (1 - alpha))
            focal_weight = jt.where(targets == 1, 1 - cls, cls)
            focal_weight = alpha_factor * jt.pow(focal_weight, gamma)

            bce = -(targets * jt.log(cls) + (1 - targets) * jt.log(1 - cls))

            # 原始分类损失 (未处理 -1 标签)
            cls_loss_raw = focal_weight * bce
            #  print(f"原始分类损失总和 (Focal * BCE, 未过滤 -1 标签): {cls_loss_raw.sum().item():.6f}")

            # 忽略 targets 为 -1 的损失
            cls_loss = jt.where(targets != -1, cls_loss_raw, jt.zeros_like(cls_loss_raw))
            #  print(f"过滤 -1 标签后的分类损失总和: {cls_loss.sum().item():.6f}")

            # 根据正样本数量归一化
            # jt.clamp(num_pos.float(), min_v = 1.0) 确保分母不为0
            cls_losses.append(cls_loss.sum() / jt.clamp(num_pos.float(), min_v = 1.0))

            # 回归损失计算 (Smooth L1 Loss)
            if pos_inds.sum() > 0:
                assigned_pos = assigned[pos_inds]
                widths_pi = widths[pos_inds]
                heights_pi = heights[pos_inds]
                ctr_x_pi = ctr_x[pos_inds]
                ctr_y_pi = ctr_y[pos_inds]

                gt_w = assigned_pos[:, 2] - assigned_pos[:, 0]
                gt_h = assigned_pos[:, 3] - assigned_pos[:, 1]
                gt_cx = assigned_pos[:, 0] + 0.5 * gt_w
                gt_cy = assigned_pos[:, 1] + 0.5 * gt_h

                gt_w = jt.clamp(gt_w, min_v = 1)
                gt_h = jt.clamp(gt_h, min_v = 1)

                targets_dx = (gt_cx - ctr_x_pi) / widths_pi
                targets_dy = (gt_cy - ctr_y_pi) / heights_pi
                targets_dw = jt.log(gt_w / widths_pi)
                targets_dh = jt.log(gt_h / heights_pi)

                reg_targets = jt.stack([targets_dx, targets_dy, targets_dw, targets_dh], dim = 1)
                reg_targets = reg_targets / jt.array([0.1, 0.1, 0.2, 0.2])  # 回归目标归一化

                diff = jt.abs(reg_targets - reg[pos_inds])
                reg_loss = jt.where(diff <= (1.0 / 9.0), 0.5 * 9.0 * jt.pow(diff, 2), diff - 0.5 / 9.0)
                reg_losses.append(reg_loss.mean())
                #  print(f"批次 {i} 回归损失: {reg_loss.mean().item():.6f}")
            else:
                reg_losses.append(jt.zeros((1,)))
                #  print(f"批次 {i} 回归损失: 0.0 (无正样本)")

        return jt.stack(cls_losses).mean(), jt.stack(reg_losses).mean()