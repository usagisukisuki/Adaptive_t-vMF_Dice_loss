#coding: utf-8
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision.models as models
import os
import argparse
import random
import sys

import utils as ut
from mydataset import CVCClinicDB_Dataset
from unet import U_Net
from loss import Adaptive_tvMF_DiceLoss
from dsc import DiceScoreCoefficient


# training #
def train(epoch, iters, kappa):
    model.train()
    sum_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        targets = targets.long()
         
        output = model(inputs)

        loss = criterion(output, targets, kappa)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        print("iter %d / %d  train_Loss: %.4f" % (iters+1, args.maxiter, loss))
        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (iters+1, loss))

        iters += 1

        if iters == args.maxiter:
            sys.exit()

        adjust_learning_rate(optimizer, iters)
        
    return sum_loss/(batch_idx+1), iters

# validation #
def test(epoch,kappa):
    model.eval()
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()
                
            output = model(inputs)

            output = F.softmax(output, dim=1)
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()
 
            for j in range(args.batchsize):
                predict.append(output[j])
                answer.append(targets[j])

        dsc = DiceScoreCoefficient(n_classes=args.classes)(predict, answer)

    return dsc

# adjust learning rate #
def adjust_learning_rate(optimizer, iters):
    lr = 0.01*(1 - iters/args.maxiter)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# adjust the parameter kappa #
def adjust_kappa(mm):
    return torch.Tensor(mm*args.alpha).cuda(device)

###### main ######
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tvMF Dice loss')
    parser.add_argument('--classes', '-c', type=int, default=2)
    parser.add_argument('--batchsize', '-b', type=int, default=24)
    parser.add_argument('--num_epochs', '-e', type=int, default=200)
    parser.add_argument('--maxiter', '-m', type=int, default=4000)
    parser.add_argument('--alpha', '-a', type=float, default=32.0)
    parser.add_argument('--dataset', '-i', default='./data')
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--gpu', '-g', type=str, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    gpu_flag = args.gpu

    # device #
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # save #
    if not os.path.exists("{}".format(args.out)):
      	os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "model")):
      	os.mkdir(os.path.join("{}".format(args.out), "model"))

    PATH_1 = "{}/trainloss.txt".format(args.out)
    PATH_2 = "{}/testloss.txt".format(args.out)
    PATH_3 = "{}/DSC.txt".format(args.out)

    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass

    # seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # preprocceing #
    train_transform = ut.ExtCompose([ut.ExtResize((224,224)),
                                     ut.ExtRandomRotation(degrees=90),
                                     ut.ExtRandomHorizontalFlip(),
                                     ut.ExtToTensor(),
                                     ])

    val_transform = ut.ExtCompose([ut.ExtResize((224,224)),
                                   ut.ExtToTensor(),
                                   ])

    # data loader #
    data_train = CVCClinicDB_Dataset(dataset_type='train', transform=train_transform) 
    data_val = CVCClinicDB_Dataset(dataset_type='val', transform=val_transform)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=args.batchsize, shuffle=False, drop_last=True, num_workers=2)

    # model #
    model = U_Net(output=args.classes).cuda(device)

    # loss function #
    criterion = Adaptive_tvMF_DiceLoss(n_classes=args.classes)

    # optimizer #
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    ### training & validation ###
    sample = 0
    sample_loss = 10000000
    iters = 0
    kappa = torch.Tensor(np.zeros((args.classes))).cuda(device)

    for epoch in range(args.num_epochs):
        loss_train, iters = train(epoch, iters, kappa)

        dsc = test(epoch, kappa)
        kappa = adjust_kappa(dsc)

        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, np.mean(dsc)))


        PATH_train ="{}/model/model_train.pth".format(args.out)
        torch.save(model.state_dict(), PATH_train)

        if np.mean(dsc) >= sample:
            sample = np.mean(dsc)
            PATH_dsc ="{}/model/model_bestdsc.pth".format(args.out)
            torch.save(model.state_dict(), PATH_dsc)

        if loss_train < sample_loss:
           sample_loss = loss_train
           PATH_loss ="{}/model/model_bestloss.pth".format(args.out)
           torch.save(model.state_dict(), PATH_loss)

        print("")
        print("Average DSC : %.4f" % np.mean(dsc))
        print("")


