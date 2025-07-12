import numpy as np
import jittor.nn as nn
import jittor as jt

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        self.pyramid_levels = pyramid_levels if pyramid_levels is not None else [3, 4, 5, 6, 7]
        self.strides = strides if strides is not None else [2 ** x for x in self.pyramid_levels]
        self.sizes = sizes if sizes is not None else [2 ** (x + 2) for x in self.pyramid_levels]

        # 直接将 ratios 和 scales 存储为 Jittor 张量
        # 确保它们在后续计算中是 Jittor 类型
        self.ratios = jt.array(ratios if ratios is not None else [0.5, 1, 2], dtype=jt.float32)
        self.scales = jt.array(scales if scales is not None else [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=jt.float32)

    def execute(self, image):
        # image_shape 从 Jittor 张量的 shape 中获取
        image_shape = image.shape[2:] # [H, W]

        # 确保 image_shapes 的计算也在 Jittor 张量上进行，或者转换为 Jittor 张量
        # 这里为了兼容性，仍然保留了部分Python整数操作，但在传递给Jittor函数时会转换
        image_shapes = []
        for x in self.pyramid_levels:
            # 使用Jittor张量进行除法和取整
            shape_h = (image_shape[0] + (2 ** x) - 1) // (2 ** x)
            shape_w = (image_shape[1] + (2 ** x) - 1) // (2 ** x)
            image_shapes.append((shape_h, shape_w))


        # 收集所有层级的 anchors，避免低效的 np.append
        all_anchors_list = []

        for idx, p in enumerate(self.pyramid_levels):
            # generate_anchors 现在返回 Jittor 张量
            anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            # shift 现在也处理并返回 Jittor 张量
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors_list.append(shifted_anchors)

        # 一次性拼接所有层级的 anchors
        all_anchors = jt.concat(all_anchors_list, dim=0)

        # 增加批次维度
        all_anchors = all_anchors.unsqueeze(0)

        # anchors 不需要梯度
        return all_anchors.stop_grad()

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window. All operations use Jittor tensors.
    """
    # 确保输入是 Jittor 张量
    if ratios is None:
        ratios = jt.array([0.5, 1, 2], dtype=jt.float32)
    if scales is None:
        scales = jt.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=jt.float32)

    num_anchors = ratios.shape[0] * scales.shape[0]

    # 初始化输出 anchors 为 Jittor 张量
    anchors = jt.zeros((num_anchors, 4), dtype=jt.float32)

    # 扩展 scales 到 (num_anchors, 2)
    # 使用 jt.repeat 替代 np.tile
    anchors[:, 2:] = base_size * scales.repeat(ratios.shape[0], 1).reshape(-1,1).repeat(1,2)

    # 计算 areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # 根据 ratios 校正
    # 使用 jt.repeat 替代 np.repeat
    anchors[:, 2] = jt.sqrt(areas / ratios.repeat(scales.shape[0]))
    anchors[:, 3] = anchors[:, 2] * ratios.repeat(scales.shape[0])

    # 从 (x_ctr, y_ctr, w, h) 转换为 (x1, y1, x2, y2)
    # 使用 jt.repeat 替代 np.tile
    anchors[:, 0::2] -= (anchors[:, 2] * 0.5).unsqueeze(1).repeat(1, 2)
    anchors[:, 1::2] -= (anchors[:, 3] * 0.5).unsqueeze(1).repeat(1, 2)

    return anchors


def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.
    :param image_shape: Jittor tensor shape (H, W) or tuple/list
    :param pyramid_levels:
    :return: list of tuples, each (H_level, W_level)
    """
    # 确保 image_shape 是一个可迭代的，并且里面的元素是整数
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shapes = []
    for x in pyramid_levels:
        h_level = (image_shape_h + (2 ** x) - 1) // (2 ** x)
        w_level = (image_shape_w + (2 ** x) - 1) // (2 ** x)
        image_shapes.append((h_level, w_level))
    return image_shapes


def anchors_for_shape(
        image_shape,
        pyramid_levels = None,
        ratios = None,
        scales = None,
        strides = None,
        sizes = None,
):
    # 确保 ratios 和 scales 是 Jittor 张量
    ratios = jt.array(ratios, dtype=jt.float32) if ratios is not None else jt.array([0.5, 1, 2], dtype=jt.float32)
    scales = jt.array(scales, dtype=jt.float32) if scales is not None else jt.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=jt.float32)

    pyramid_levels = pyramid_levels if pyramid_levels is not None else [3, 4, 5, 6, 7]
    strides = strides if strides is not None else [2 ** x for x in pyramid_levels]
    sizes = sizes if sizes is not None else [2 ** (x + 2) for x in pyramid_levels]

    image_shapes = compute_shape(image_shape, pyramid_levels)

    all_anchors_list = []
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors_list.append(shifted_anchors)

    return jt.concat(all_anchors_list, dim=0)


def shift(shape, stride, anchors):
    # 将 shape 和 stride 转换为 Jittor 张量，以确保后续 Jittor 操作
    # shape 通常是 (H, W) 的 tuple/list，这里直接取元素
    shift_x = (jt.arange(0, shape[1]).float32() + 0.5) * float(stride)
    shift_y = (jt.arange(0, shape[0]).float32() + 0.5) * float(stride)

    shift_y_grid, shift_x_grid = jt.meshgrid(shift_y, shift_x) # Jittor meshgrid 默认 (rows, cols) 顺序

    # stack 替代 vstack 和 transpose
    shifts = jt.stack((
        shift_x_grid.flatten(), shift_y_grid.flatten(),
        shift_x_grid.flatten(), shift_y_grid.flatten()
    ), dim=1)

    A = anchors.shape[0] # number of anchors (e.g., 9)
    K = shifts.shape[0]  # number of feature map locations

    # 使用 Jittor 的广播机制来添加锚框
    # anchors: (1, A, 4)
    # shifts:  (K, 1, 4)
    all_anchors = (anchors.unsqueeze(0) + shifts.unsqueeze(1))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors
