# Adaptive t-vMF Dice Loss
This repository is the official PyTorch implementation for our Computers in Biology and Medicine (CBM) 2023 paper ''Adaptive t-vMF dice loss: An effective expansion of dice loss for medical image segmentation'' [[paper]](https://www.sciencedirect.com/science/article/pii/S0010482523011605) (**impact factor=7.7**).

## Highlights
<div align="center">
  <img src="figs/git_fig4.png" width="80%">
</div>

- The Dice loss is able to rewrite in the loss function using the **cosine similarity**.
- **T-vMF Dice loss** is formulated in a more compact similarity than the Dice loss.
- **Adaptive t-vMF Dice loss** is able to use more compact similarities for easy classes and wider similarities for difficult classes.
- Our loss functions can achieve high performance in various type of datasets in spite of set only one parameter decided by human.


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


