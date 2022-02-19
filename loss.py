import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F

##### tvMF Dice loss #####
class tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes, kappa=None):
        super(tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.kappa = kappa

    ### one-hot encoding ###
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    ### tvmf dice loss ###
    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        smooth = 1.0

        score = F.normalize(score, p=2, dim=[0,1,2])
        target = F.normalize(target, p=2, dim=[0,1,2])
        cosine = torch.sum(score * target)
        intersect =  (1. + cosine).div(1. + (1.- cosine).mul(kappa)) - 1.
        loss = (1 - intersect)**2.0

        return loss

    ### main ###
    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0

        for i in range(0, self.n_classes):
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], self.kappa)
            loss += tvmf_dice
        return loss / self.n_classes


##### Adaptive tvMF Dice loss #####
class Adaptive_tvMF_DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(Adaptive_tvMF_DiceLoss, self).__init__()
        self.n_classes = n_classes

    ### one-hot encoding ###
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    ### tvmf dice loss ###
    def _tvmf_dice_loss(self, score, target, kappa):
        target = target.float()
        smooth = 1.0

        score = F.normalize(score, p=2, dim=[0,1,2])
        target = F.normalize(target, p=2, dim=[0,1,2])
        cosine = torch.sum(score * target)
        intersect =  (1. + cosine).div(1. + (1.- cosine).mul(kappa)) - 1.
        loss = (1 - intersect)**2.0

        return loss

    ### main ###
    def forward(self, inputs, target, kappa=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0

        for i in range(0, self.n_classes):
            tvmf_dice = self._tvmf_dice_loss(inputs[:, i], target[:, i], kappa[i])
            loss += tvmf_dice
        return loss / self.n_classes
