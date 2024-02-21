import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn

##### Binary segmentation #####
class TverskyLoss_Binary(nn.Module):
    def __init__(self, alpha=0.7):
        super(TverskyLoss_Binary, self).__init__()
        self.alpha = alpha
        self.beta = 1 - self.alpha

    def _tversky_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        TP = torch.sum(score * target)
        FP = torch.sum((1 - target) * score)
        FN = torch.sum(target * (1 - score))

        loss = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  

        return 1 - loss

    def forward(self, inputs, target, sigmoid=True):
        if sigmoid:
            inputs = torch.sigmoid(inputs)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = self._tversky_loss(inputs[:, 0], target[:, 0])

        return loss 

##### Multi-class segmentation #####
class TverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.7):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = 1 - self.alpha

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tversky_loss(self, score, target):
        target = target.float()
        smooth = 1.0

        TP = torch.sum(score * target)
        FP = torch.sum((1 - target) * score)
        FN = torch.sum(target * (1 - score))

        loss = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  

        return 1 - loss

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._tversky_loss(inputs[:, i], target[:, i])

        return loss / self.n_classes
