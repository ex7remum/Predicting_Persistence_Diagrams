import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.eps = epsilon

    def forward(self, batch, model_out):
        pred_mask = model_out['pred_mask']
        real_mask = batch['seg_mask']
        enum = 2 * (pred_mask * real_mask).sum(dim=3).sum(dim=2) # (bs x num_diags)
        denum = (pred_mask**2).sum(dim=3).sum(dim=2) + (real_mask**2).sum(dim=3).sum(dim=2) + self.eps #(bs x num_diags)
        batched_res = (1 - enum / denum).sum(dim=1)
        return batched_res.mean()
