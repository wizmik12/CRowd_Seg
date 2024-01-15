import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
# =======================================

from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

eps=1e-7

def noisy_label_loss(pred, cms, labels, ignore_index, min_trace = False, alpha=0.1, loss_mode=None):
    """ Loss for the crowdsourcing methods
    """
    b, c, h, w = pred.size()

    #
    pred_norm = pred.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)
    cm = cms.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
    cm = cm / cm.sum(1, keepdim=True)

    pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
    pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

    if loss_mode == 'ce':
        loss_ce = nn.NLLLoss(reduction='mean', ignore_index=ignore_index)(torch.log(pred_noisy+eps), labels.view(b, h, w).long())
    elif loss_mode == 'dice':
        loss_ce = DiceLoss(ignore_index=ignore_index, from_logits=False, mode='multiclass')(pred_noisy, labels.view(b, h, w).long())
    elif loss_mode == 'focal':
        loss_ce = FocalLoss(reduction='mean', ignore_index=ignore_index, mode='multiclass')(pred_noisy, labels.view(b, h, w).long())

    # regularization
    regularisation = torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
    regularisation = alpha * regularisation

    if min_trace:
        loss = loss_ce + regularisation
    else:
        loss = loss_ce - regularisation

    return loss, loss_ce, regularisation
