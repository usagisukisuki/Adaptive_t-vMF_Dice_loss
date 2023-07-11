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
from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from FCBFormer import models
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

    img.save("{}_{}/Image/Inputs/{}.png".format(args.out, args.model, path), quality=95)
    dst1.save("{}_{}/Image/Seg/{}.png".format(args.out, args.model, path), quality=95)
    dst2.save("{}_{}/Image/Ano/{}.png".format(args.out, args.model, path), quality=95)

### test ###
def test():
    model_path = "{}_{}/model/model_bestdsc.pth".format(args.out, args.model)
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
    parser.add_argument('--path', '-i', default='./data/Kvasir-SEG')
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--models', '-mo', type=str, default='unet')
    parser.add_argument('--gpu', '-g', type=str, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    gpu_flag = args.gpu

    # device #
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # save #
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.model), "Image")):
      	os.mkdir(os.path.join("{}_{}".format(args.out, args.model), "Image"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.model), "Image", "Inputs")):
      	os.mkdir(os.path.join("{}_{}".format(args.out, args.model), "Image", "Inputs"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.model), "Image", "Seg")):
      	os.mkdir(os.path.join("{}_{}".format(args.out, args.model), "Image", "Seg"))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.model), "Image", "Ano")):
      	os.mkdir(os.path.join("{}_{}".format(args.out, args.model), "Image", "Ano"))

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
    data_test = CVCClinicDB_Dataset(root = args.path, 
                                    dataset_type='test', 
                                    transform=test_transform)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batchsize, shuffle=False, drop_last=True, num_workers=2)

    # model #
    if args.model=='unet':
        model = U_Net(output=args.classes).cuda(device)

    elif args.model=='trans_unet':
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = args.classes
        config_vit.n_skip = 3
        if "R50-ViT-B_16".find('R50') != -1:
            config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        model = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes).cuda(device)
        model.load_from(weights=np.load(config_vit.pretrained_path))

    elif args.model=='fcb':
        model = models.FCBFormer(output=args.classes).cuda(device)

    ### test ###
    test()



