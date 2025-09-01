import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def np_local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def downscale_label_ratio(gt,
                          scale_factor,
                          min_ratio,
                          n_classes,
                          ignore_index=255):
    assert scale_factor > 1
    bs, orig_c, orig_h, orig_w = gt.shape
    assert orig_c == 1
    trg_h, trg_w = orig_h // scale_factor, orig_w // scale_factor
    ignore_substitute = n_classes

    out = gt.clone()  # otw. next line would modify original gt
    out[out == ignore_index] = ignore_substitute
    out = F.one_hot(
        out.squeeze(1), num_classes=n_classes + 1).permute(0, 3, 1, 2)
    assert list(out.shape) == [bs, n_classes + 1, orig_h, orig_w], out.shape
    out = F.avg_pool2d(out.float(), kernel_size=scale_factor)
    gt_ratio, out = torch.max(out, dim=1, keepdim=True)
    out[out == ignore_substitute] = ignore_index
    out[gt_ratio < min_ratio] = ignore_index
    assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
    return out

class AdaptiveInstanceNormalization(nn.Module):

    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def forward(self, x_cont, x_style=None):
        if x_style is not None:
            assert (x_cont.size()[:2] == x_style.size()[:2])
            size = x_cont.size()
            style_mean, style_std = calc_mean_std(x_style)
            content_mean, content_std = calc_mean_std(x_cont)

            normalized_x_cont = (x_cont - content_mean.expand(size))/content_std.expand(size)
            denormalized_x_cont = normalized_x_cont * style_std.expand(size) + style_mean.expand(size)
            # print('wild:', style_mean.expand(size), style_std.expand(size))
            # print('source:', content_mean.expand(size), content_std.expand(size))
            # print('x :', x_cont)
            # print('-----------------')
            # print('out :', denormalized_x_cont)
            return denormalized_x_cont

        else:
            return x_cont


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std