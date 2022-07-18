# Adaptive t-vMF Dice Loss
This repository is the official PyTorch implementation ''Adaptive t-vMF Dice Loss for Multi-class Medical Image Segmentation''.

## Introduction
<div align="center">
  <img src="figs/git_fig1.png" width="80%">
</div>
Dice loss is widely used for medical image segmentation, and many improvement loss functions based on such loss have been proposed. However, further Dice loss improvements are still possible. In this study, we reconsidered the use of Dice loss and discovered that Dice loss can be rewritten in the loss function using the cosine similarity through a simple equation transformation. Using this knowledge, we present a novel t-vMF Dice loss based on the t-vMF similarity instead of the cosine similarity. Based on the t-vMF similarity, our proposed Dice loss is formulated in a more compact similarity loss function than the original Dice loss. Furthermore, we present an effective algorithm that automatically determines the parameter $\kappa$ for the t-vMF similarity using a validation accuracy, called Adaptive t-vMf Dice loss. Using this algorithm, it is possible to apply more compact similarities for easy classes and wider similarities for difficult classes, and we are able to achieve an adaptive training based on the accuracy of the class. Through experiments conducted on four datasets using a five-fold cross validation, we confirmed that the Dice score coefficient (DSC) was further improved in comparison with the original Dice loss and other loss functions.
<br />
<br />
In this repository, we have prepared CVC-ClinicDB dataset and the code of our paper.

## Preparation for CVC-ClinicDB dataset
Please download from [[CVC-ClinicDB]](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) and extract them under $/Dataset, and make them look like this:
```
lite_hrnet
├── configs
├── models
├── tools
`── data
    │── coco
        │-- annotations
        │   │-- person_keypoints_train2017.json
        │   |-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        │-- train2017
        │   │-- 000000000009.jpg
        │   │-- 000000000025.jpg
        │   │-- 000000000030.jpg
        │   │-- ...
        `-- val2017
            │-- 000000000139.jpg
            │-- 000000000285.jpg
            │-- 000000000632.jpg
            │-- ...
```


## Training
If you prepared the dataset, you can directly run the following code to train the model.
```
sh train.sh
```

## Testing
If you generated the pretrain model, you can run the following code to evaluate the model.
```
sh test.sh
```

## Results and Visualization on Synaps multi-organ dataset
<div align="center">
  <img src="figs/git_fig2.png" width="100%">
</div>

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


