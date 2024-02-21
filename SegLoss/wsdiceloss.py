import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn

##### Binary segmentation #####
class WSDiceLoss_Binary(nn.Module):
    def __init__(self, v1=0.85, v2=0.15):
        super(WSDiceLoss_Binary, self).__init__()
        self.v1 = v1
        self.v2 = v2

    def _wsdice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        w = target * (self.v2 - self.v1) + self.v1
        score = w*(2.0 * score - 1.0)
        target = w*(2.0 * target - 1.0)

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        loss = (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)

        return 1 - loss


    def forward(self, inputs, target, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = self._wsdice_loss(inputs[:, 0], target[:, 0])

        return loss


##### Multi-class segmentation #####
class WSDiceLoss(nn.Module):
    def __init__(self, n_classes, v1=0.85, v2=0.15):
        super(WSDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.v1 = v1
        self.v2 = v2

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _wsdice_loss(self, score, target):
        target = target.float()
        smooth = 1.0
        w = target * (self.v2 - self.v1) + self.v1
        score = w*(2.0 * score - 1.0)
        target = w*(2.0 * target - 1.0)

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        loss = (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)

        return 1 - loss


    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._wsdice_loss(inputs[:, i], target[:, i])

        return loss / self.n_classes