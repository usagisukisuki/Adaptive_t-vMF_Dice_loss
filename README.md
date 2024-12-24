# :game_die: Adaptive t-vMF Dice Loss :game_die:
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unmasking-anomalies-in-road-scene/anomaly-detection-on-lost-and-found)](https://paperswithcode.com/sota/medical-image-segmentation-on-cvc-clinicdb)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unmasking-anomalies-in-road-scene/anomaly-detection-on-lost-and-found)](https://paperswithcode.com/sota/medical-image-segmentation-on-automatic)

This repository is the official PyTorch implementation for our **Computers in Biology and Medicine (CBM)** paper ''Adaptive t-vMF dice loss: An effective expansion of dice loss for medical image segmentation'' [[paper]](https://www.sciencedirect.com/science/article/pii/S0010482523011605) (**impact factor=7.7**).

## :game_die: Highlights
<div align="center">
  <img src="figs/git_fig4.png" width="80%">
</div>

- The Dice loss is able to rewrite in the loss function using the **cosine similarity**.
- **T-vMF Dice loss** is formulated in a more compact similarity than the Dice loss.
- **Adaptive t-vMF Dice loss** is able to use more compact similarities for easy classes and wider similarities for difficult classes.
- Our loss functions can achieve high performance in various type of datasets in spite of set only one parameter decided by human.


## :game_die: Datasets
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

## :game_die: Checkpoints
Please download pre-trained models for TransUNet and FCBFormer encoders from [[TransUNet]](https://github.com/Beckschen/TransUNet), and [[FCBFormer]](https://github.com/ESandML/FCBFormer).

## :game_die: Training
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

### Using other loss functions
Please use the source code in the **SegLoss**:
```
SegLoss
├── diceloss.py
├── focal_tverskyloss.py
├── noise_robust_diceloss.py
├── ...

```

## :game_die: Inference
If you generated the pretrain model, you can run the following code to evaluate the model.
```
sh test.sh
```


## :game_die: Results and Visualization
We assessed 4 datasets and applied the Dice Score Coefficient (DSC) to evaluate the segmentation accuracy.
|Loss|CVC-ClinicDB|Kvasior-SEG|ACDC|Synapse|
|:---:|:---:|:---:|:---:|:---:|
|Cross Entropy (CE)|81.83|89.10|90.74|50.59|
|Dice|82.02|90.34|92.21|70.57|
|CE+Dice|83.45|91.25|92.48|71.49|
|Focal+Dice|82.25|90.19|92.32|70.55|
|Generalized Dice|83.37|88.70|92.42|66.53|
|Noise-robust Dice|83.14|90.34|92.40|64.74|
|Focal Dice|84.38|90.98|92.82|68.97|
|Focal-Tversky|83.39|90.61|92.44|64.41|
|Adaptive t-VMF Dice (Ours)|**88.68**|**92.24**|**93.68**|**74.22**|

<div align="center">
  <img src="figs/git_fig3.png" width="100%">
</div>

## :game_die: Citation
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


