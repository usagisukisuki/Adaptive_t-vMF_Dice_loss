import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn

##### Binary segmentation #####
class Dice_FocalLoss_Binary(nn.Module):
    def __init__(self, lamda=0.5, alpha=0.5):
        super(Dice_FocalLoss_Binary, self).__init__()
        self.lamda = lamda
        self.alpha = alpha
        self.beta = 1 - self.alpha

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum(score * target)
        fn = torch.sum((1 - score) * target)
        fp = torch.sum(score * (1 - target))
        
        loss = (intersect + smooth) / (intersect + self.alpha*fn + self.beta*fp + smooth)

        return loss

    def _focal_loss(self, score, target):
        target = target.float()
        smooth = 1e-7
        focal = (1 - score)**2.0
        loss = focal * target * torch.log(score + smooth)

        return loss.mean()

    def forward(self, inputs, target, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        dloss += self._dice_loss(inputs[:, 0], target[:, 0])
        floss = self._focal_loss(inputs[:, 0], target[:, 0])

        return (1 - dloss) - self.lamda * floss


##### Multi-class segmentation #####
class Dice_FocalLoss(nn.Module):
    def __init__(self, n_classes, lamda=0.5, alpha=0.5):
        super(Dice_FocalLoss, self).__init__()
        self.n_classes = n_classes
        self.lamda = lamda
        self.alpha = alpha
        self.beta = 1 - self.alpha

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum(score * target)
        fn = torch.sum((1 - score) * target)
        fp = torch.sum(score * (1 - target))
        
        loss = (intersect + smooth) / (intersect + self.alpha*fn + self.beta*fp + smooth)

        return loss

    def _focal_loss(self, score, target):
        target = target.float()
        smooth = 1e-7
        focal = (1 - score)**2.0
        loss = focal * target * torch.log(score + smooth)

        return loss.mean()

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        bceloss = 0.0
        for i in range(0, self.n_classes):
            dloss += self._dice_loss(inputs[:, i], target[:, i])
            floss += self._focal_loss(inputs[:, i], target[:, i])

        return (self.n_classes - dloss) - self.lamda * floss