# DKS-DoubleU-Net
![]( https://img.shields.io/badge/license-MIT-green.svg)  
This repo. is the official implementation of "Segmentation of Breast Tubules in H&E Images Based on a DKS-DoubleU-Net Model" .  
Please see the [paper](https://www.hindawi.com/journals/bmri/2022/2961610/).  

## Overview    
![tupian](/images/DKS-DoubleU-Net.png)

## DenseNet    
![tupian](/images/densenet.png)

## Kernel Selecting Module    
![tupian](/images/KSM.png)

## Run  
1.Requirements:  
* python3  
* tensorflow 2.3.0  
* keras 1.0.8  

We have uploaded the corresponding environment package for your convenience.  

2.Training:  
* Prepare the required images and store them in new_data folder, the example format of the training, validation, and testing datasets is in the  "new_data" folder.
* Run ``` python train.py```  

3.Testing:
* Run ```python predict.py```

## Citation  
If you find our paper/code is helpful, please consider citing:  
```Yuli Chen, Yao Zhou, Guoping Chen, Yuchuan Guo, Yanquan Lv, Miao Ma, Zhao Pei, Zengguo Sun, "Segmentation of Breast Tubules in H&E Images Based on a DKS-DoubleU-Net Model", BioMed Research International, vol. 2022, Article ID 2961610, 12 pages, 2022. https://doi.org/10.1155/2022/2961610```



