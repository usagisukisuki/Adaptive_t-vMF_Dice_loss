# Adaptive t-vMF Dice Loss
This repository is the official PyTorch implementation for our Computers in Biology and Medicine (CBM) 2023 paper ''Adaptive t-vMF dice loss: An effective expansion of dice loss for medical image segmentation'' [[paper]](https://www.sciencedirect.com/science/article/pii/S0010482523011605) (**impact factor=7.7**).

## Introduction
<div align="center">
  <img src="figs/git_fig4.png" width="80%">
</div>
Dice loss is widely used for medical image segmentation, and many improved loss functions have been proposed. However, further Dice loss improvements are still possible. In this study, we reconsidered the use of Dice loss and discovered that Dice loss can be rewritten in the loss function using the cosine similarity through a simple equation transformation. Using this knowledge, we present a novel t-vMF Dice loss based on the t-vMF similarity instead of the cosine similarity. Based on the t-vMF similarity, our proposed Dice loss is formulated in a more compact similarity loss function than the original Dice loss. Furthermore, we present an effective algorithm that automatically determines the parameter 
 for the t-vMF similarity using a validation accuracy, called Adaptive t-vMF Dice loss. Using this algorithm, it is possible to apply more compact similarities for easy classes and wider similarities for difficult classes, and we are able to achieve adaptive training based on the accuracy of each class. We evaluated binary segmentation datasets of CVC-ClinicDB and Kvasir-SEG, and multi-class segmentation datasets of Automated Cardiac Diagnosis Challenge and Synapse multi-organ segmentation. Through experiments conducted on four datasets using a five-fold cross-validation, we confirmed that the Dice score coefficient (DSC) was further improved in comparison with the original Dice loss and other loss functions.
<br />
<br />

## Preparation for preprocessing datasets
Please download from [[Dataset]](https://drive.google.com/drive/folders/1q80fDpAM62jPR5p61_4BVzCX4I1KZMqt?usp=drive_link) and extract them under "data", and make them look like this:
```
data
├── CVC-ClinicDB
    ├── datamodel
        ├── train_data_1.npy
        ├── train_label_1.npy
        ├── ...
├── Kvasir-SEG
├── ACDC
`── Synapse

```

## Pre-trained model for TransUNet and FCBFormer
Please download pre-trained models for TransUNet and FCBFormer encoders from [[TransUNet]](https://github.com/Beckschen/TransUNet), and [[FCBFormer]](https://github.com/ESandML/FCBFormer).

## Training
### t-vMF Dice loss
If you prepared the dataset, you can directly run the following code to train the model.
```
python3 train.py -g 0 -o result -e 200 -b 24 -s 0 -mo unet -lo tvmf -c 2
```
### Adaptive t-vMF Dice loss
If you prepared the dataset, you can directly run the following code to train the model.
```
python3 train.py -g 0 -o result -e 200 -b 24 -s 0 -mo unet -lo Atvmf -c 2
```

## Testing
If you generated the pretrain model, you can run the following code to evaluate the model.
```
sh test.sh
```

## Using other loss functions
Please use the source code in the **SegLoss**:
```
SegLoss
├── diceloss.py
├── focal_tverskyloss.py
├── noise_robust_diceloss.py
├── ...

```


## Results and Visualization
<div align="center">
  <img src="figs/git_fig3.png" width="100%">
</div>

## Citation
```
@article{kato2023adaptive,
  title={Adaptive t-vMF dice loss: An effective expansion of dice loss for medical image segmentation},
  author={Kato, Sota and Hotta, Kazuhiro},
  journal={Computers in Biology and Medicine},
  pages={107695},
  year={2023},
  publisher={Elsevier}
}
```


