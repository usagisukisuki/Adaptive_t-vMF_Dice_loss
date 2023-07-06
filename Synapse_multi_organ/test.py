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
from PIL import Image
import os
import argparse
import random
import sys

import utils as ut
from mydataset import CVCClinicDB_Dataset
from unet import U_Net
from loss import Adaptive_tvMF_DiceLoss
from dsc import DiceScoreCoefficient


### save images ###
def Save_image(img, seg, ano, path):
    seg = np.argmax(seg, axis=1)
    img = img[0]
    img = np.transpose(img, (1,2,0))
    seg = seg[0]
    ano = ano[0]
    dst1 = np.zeros((seg.shape[0], seg.shape[1],3))
    dst2 = np.zeros((seg.shape[0], seg.shape[1],3))

    #class1 : background
    #class0 : polyp

    dst1[seg==0] = [0.0, 0.0, 0.0]
    dst1[seg==1] = [255.0, 255.0, 255.0]
    dst2[ano==0] = [0.0, 0.0, 0.0]
    dst2[ano==1] = [255.0, 255.0, 255.0]

    img = Image.fromarray(np.uint8(img*255.0))
    dst1 = Image.fromarray(np.uint8(dst1))
    dst2 = Image.fromarray(np.uint8(dst2))

    img.save("{}/Image/Inputs/{}.png".format(args.out, path), quality=95)
    dst1.save("{}/Image/Seg/{}.png".format(args.out, path), quality=95)
    dst2.save("{}/Image/Ano/{}.png".format(args.out, path), quality=95)

### test ###
def test():
    model_path = "{}/model/model_bestdsc.pth".format(args.out)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()
    
            output = model(inputs)

            output = F.softmax(output, dim=1)
            inputs = inputs.cpu().numpy()
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()

            for j in range(args.batchsize):
                predict.append(output[j])
                answer.append(targets[j])

            Save_image(inputs, output, targets, batch_idx+1)

        dsc = DiceScoreCoefficient(n_classes=args.classes)(predict, answer)

        print("Dice")
        print("class 0  = %f" % (dsc[0]))
        print("class 1  = %f" % (dsc[1]))
        print("mDice     = %f" % (np.mean(dsc)))
        
        with open(PATH, mode = 'a') as f:
            f.write("%f\t%f\t%f\n" % (dsc[0], dsc[1], np.mean(dsc)))


###### main ######
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tvMF Dice loss')
    parser.add_argument('--classes', '-c', type=int, default=2)
    parser.add_argument('--batchsize', '-b', type=int, default=24)
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
    if not os.path.exists(os.path.join("{}".format(args.out), "Image")):
      	os.mkdir(os.path.join("{}".format(args.out), "Image"))
    if not os.path.exists(os.path.join("{}".format(args.out), "Image", "Inputs")):
      	os.mkdir(os.path.join("{}".format(args.out), "Image", "Inputs"))
    if not os.path.exists(os.path.join("{}".format(args.out), "Image", "Seg")):
      	os.mkdir(os.path.join("{}".format(args.out), "Image", "Seg"))
    if not os.path.exists(os.path.join("{}".format(args.out), "Image", "Ano")):
      	os.mkdir(os.path.join("{}".format(args.out), "Image", "Ano"))

    PATH = "{}/predict.txt".format(args.out)

    with open(PATH, mode = 'w') as f:
        pass


    # seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # preprocceing #
    test_transform = ut.ExtCompose([ut.ExtResize((224,224)),
                                    ut.ExtToTensor(),
                                   ])
    # data loader #
    data_test = CVCClinicDB_Dataset(dataset_type='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batchsize, shuffle=False, drop_last=True, num_workers=2)

    # model #
    model = U_Net(output=args.classes).cuda(device)

    ### test ###
    test()



