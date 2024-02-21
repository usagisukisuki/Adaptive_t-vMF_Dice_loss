import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn

##### Binary segmentation #####
class GDiceLoss_Binary(nn.Module):
    def __init__(self):
        super(GDiceLoss_Binary, self).__init__()

    def _gdice_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        weight = 1.0 / ((target.sum()*target.sum())+smooth)
        intersect = torch.sum(score * target)
        intersect = intersect * weight
        dominator = torch.sum(score + target)
        dominator = dominator * weight + smooth

        return intersect, dominator 

    def forward(self, inputs, target, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        intersect, dominator = self._gdice_loss(inputs[:, 0], target[:, 0])

        return 1 - 2*(intersect/dominator)


##### Multi-class segmentation #####
class GDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(GDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _gdice_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        weight = 1.0 / ((target.sum()*target.sum())+smooth)
        intersect = torch.sum(score * target)
        intersect = intersect * weight
        dominator = torch.sum(score + target)
        dominator = dominator * weight + smooth

        return intersect, dominator 

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        inte = 0.0
        dom = 0.0
        for i in range(0, self.n_classes):
            intersect, dominator = self._gdice_loss(inputs[:, i], target[:, i])
            inte += intersect
            dom += dominator
        return 1 - 2*(inte/dom)
