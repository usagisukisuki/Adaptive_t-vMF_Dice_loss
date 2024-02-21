import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn

##### Binary segmentation #####
class NDiceLoss_Binary(nn.Module):
    def __init__(self, gamma=1.5):
        super(NDiceLoss_Binary, self).__init__()
        self.gamma = gamma


    def _ndice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum((torch.abs(score - target))**self.gamma)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (intersect + smooth) / (z_sum + y_sum + smooth)

        return loss

    def forward(self, inputs, target, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = self._ndice_loss(inputs[:, 0], target[:, 0])

        return loss

##### Multi-class segmentation #####
class NDiceLoss(nn.Module):
    def __init__(self, n_classes, gamma=1.5):
        super(NDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.gamma = gamma

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _ndice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        intersect = torch.sum((torch.abs(score - target))**self.gamma)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (intersect + smooth) / (z_sum + y_sum + smooth)

        return loss

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._ndice_loss(inputs[:, i], target[:, i])

        return loss / self.n_classes
