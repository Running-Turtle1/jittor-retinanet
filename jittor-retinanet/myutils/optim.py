# import jittor as jt
# def clip_grad_norm(self, max_norm: float, norm_type: int = 2):
#     """剪切此优化器的梯度范数，范数是对所有梯度一起计算的。
#
#     参数：
#         max_norm (``float`` or ``int``): 梯度的最大范数
#         norm_type (``int``): 1-范数或2-范数
#
#     示例：
#        >>> a = jt.ones(2)
#        ... opt = jt.optim.SGD([a], 0.1)
#        ... loss = a*a
#        ... opt.zero_grad()
#        ... opt.backward(loss)
#        ... print(opt.param_groups[0]['grads'][0].norm())
#        2.83
#        >>> opt.clip_grad_norm(0.01, 2)
#        ... print(opt.param_groups[0]['grads'][0].norm())
#        0.01
#        >>> opt.step()
#     """
#     if self.__zero_grad: return
#     grads = []
#     for pg in self.param_groups:
#         for p, g in zip(pg["params"], pg["grads"]):
#             if p.is_stop_grad(): continue
#             grads.append(g.flatten())
#     if len(grads) == 0: return
#     total_norm = jt.norm(jt.concat(grads), norm_type)
#     clip_coef = jt.minimum(max_norm / (total_norm + 1e-6), 1.0)
#     for pg in self.param_groups:
#         for p, g in zip(pg["params"], pg["grads"]):
#             if p.is_stop_grad(): continue
#             g.update(g * clip_coef)
# myutils/optim.py (修正并补充后的完整文件)
import jittor as jt

def clip_grad_norm_(parameters, max_norm: float, norm_type: int = 2):
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[jt.Var]): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (int): type of the used p-norm. Can be 'inf' for
            infinity norm.
    """
    if isinstance(parameters, jt.Var):
        parameters = [parameters]

    # 过滤掉不需要梯度的参数
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return jt.core.Var(0.0)

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    # 计算总范数
    total_norm = jt.norm(jt.stack([g.norm(norm_type) for g in grads]), norm_type)

    # 计算裁剪系数
    clip_coef = max_norm / (total_norm + 1e-6)

    # 如果范数未超过最大值，则不裁剪
    if clip_coef < 1:
        for g in grads:
            g.assign(g * clip_coef)

    return total_norm