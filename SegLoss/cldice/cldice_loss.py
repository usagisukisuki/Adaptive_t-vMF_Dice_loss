import torch
import torch.nn as nn
import torch.nn.functional as F
from soft_skel import soft_skel


class Soft_Dice_clDice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5):
        super(Soft_Dice_clDice, self).__init__()
        self.iter = iter_
        self.alpha = alpha

    def soft_dice(self, target, score):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

        return 1 - loss

    def cl_dice(self, y_pred, skel_pred, y_true, skel_true):
        y_true = y_true.float()
        skel_true = skel_true.float()
        smooth = 1.0

        tprec = (torch.sum(skel_pred * y_true)+smooth)/(torch.sum(skel_pred)+smooth)    
        tsens = (torch.sum(skel_true * y_pred)+smooth)/(torch.sum(skel_true)+smooth)    
        cl_dice = (2.0*(tprec*tsens))/(tprec+tsens)

        return 1 - cl_dice

    def forward(self, y_true, y_pred, weight=None, sigmoid=True):
        if sigmoid:
            y_pred = torch.sigmoid(y_pred)

        assert y_pred.size() == y_true.size(), 'predict {} & target {} shape do not match'.format(y_pred.size(), y_true.size())

        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)

        dice = self.soft_dice(y_true[:, 0], y_pred[:,0])
        cl_dice = self.cl_dice(y_pred[:, 0], skel_pred[:, 0], y_true[:, 0], skel_true[:, 0])
        loss = (1.0-self.alpha)*dice+self.alpha*cl_dice

        return loss


