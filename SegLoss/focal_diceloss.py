import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn

##### Binary segmentation #####
class FocalDiceLoss_Binary(nn.Module):
    def __init__(self, beta=2.0):
        super(FocalDiceLoss_Binary, self).__init__()
        self.beta = 1.0 / beta

    def _fdice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

        loss = 1 - loss**self.beta

        return loss

    def forward(self, inputs, target, weight=None, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        dice = self._dice_loss(inputs[:, 0], target[:, 0])
        loss = dice * weight

        return loss 

##### Multi-class segmentation #####
class FocalDiceLoss(nn.Module):
    def __init__(self, n_classes, beta=2.0):
        super(FocalDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.beta = 1.0 / beta

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _fdice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)

        loss = 1 - loss**self.beta

        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._fdice_loss(inputs[:, i], target[:, i])
            loss += dice * weight

        return loss / self.n_classes
