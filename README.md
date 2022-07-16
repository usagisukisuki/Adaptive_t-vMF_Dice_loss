# Adaptive t-vMF Dice Loss
This repository is the official PyTorch implementation ''Adaptive t-vMF Dice Loss for Multi-class Medical Image Segmentation''.

## Introduction
We propose an automatic preprocessing and ensemble learning for segmentation of cell images with low quality. It is difficult to capture cells with strong light. Therefore, the microscopic images of cells tend to have low image quality but these images are not good for semantic segmentation. Here we propose a method to translate an input image to the images that are easy to recognize by deep learning. The proposed method consists of two deep neural networks. The first network is the usual training for semantic segmentation, and penultimate feature maps of the first network are used as filters to translate an input image to the images that emphasize each class. This is the automatic preprocessing and translated cell images are easily classified. The input cell image with low quality is translated by the feature maps in the first network, and the translated images are fed into the second network for semantic segmentation. Since the outputs of the second network are multiple segmentation results, we conduct the weighted ensemble of those segmentation images. Two networks are trained by end-to-end manner, and we do not need to prepare images with high quality for the translation.
<br />
<br />
In this repository, we have prepared the iRPE cell image dataset and the code of our paper.

## Training
If you prepared the dataset, you can directly run the following code to train the model.
```
sh train.sh
```

## Testing
If you generated the pretrain model, you can run the following code to evaluate the model.
```
sh teset.sh
```

## Citation
```
@INPROCEEDINGS{9658801,
  author={Kato, Sota and Hotta, Kazuhiro},
  booktitle={2021 IEEE International Conference on Systems, Man, and Cybernetics (SMC)}, 
  title={Automatic Preprocessing and Ensemble Learning for Cell Segmentation with Low Quality}, 
  year={2021},
  volume={},
  number={},
  pages={1836-1841},
  doi={10.1109/SMC52423.2021.9658801}
}
```


