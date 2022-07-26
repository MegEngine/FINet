# [AAAI 2022] FINet: Dual Branches Feature Interaction for Partial-to-Partial Point Cloud Registration

<h3 align = "center">Hao Xu$^{1,2}$, Nianjin Ye$^2$, Guanghui Liu$^1$, Bing Zeng$^1$, Shuaicheng Liu$^{1,2}$</h3>
<h3 align = "center"> $^1$ University of Electronic Science and Technology of China</center></h3>
<h3 align = "center">$^2$ Megvii Research</h3>
                     
                       



This is the official implementation (MegEngine implementation) of our AAAI2022 paper [FINet](https://www.aaai.org/AAAI22Papers/AAAI-549.XuH.pdf).

Our presentation video: [[Youtube](https://www.youtube.com/watch?v=XDmE9iSx9WM)][[Bilibili](https://www.bilibili.com/video/BV1z44y1s7up/)].

## Our Poster

![image](./images/FINet_poster.png)

## Dependencies

* MegEngine==1.7.0
* Other requirements please refer to`requirements.txt`.

## Data Preparation

Following [OMNet](https://github.com/megvii-research/OMNet), we use the OS and TS data of the ModelNet40 dataset.

### OS data

We refer the original data from PointNet as OS data, where point clouds are only sampled once from corresponding CAD models. We offer two ways to use OS data, (1) you can download this data from its original link [original_OS_data.zip](http://modelnet.cs.princeton.edu/). (2) you can also download the data that has been preprocessed by us from link [our_OS_data.zip](https://drive.google.com/file/d/1rXnbXwD72tkeu8x6wboMP0X7iL9LiBPq/view?usp=sharing).

### TS data

Since OS data incurs over-fitting issue, we propose our TS data, where point clouds are randomly sampled twice from CAD models. You need to download our preprocessed ModelNet40 dataset first, where 8 axisymmetrical categories are removed and all CAD models have 40 randomly sampled point clouds. The download link is [TS_data.zip](https://drive.google.com/file/d/1DPBBI3Ulvp2Mx7SAZaBEyvADJzBvErFF/view?usp=sharing). All 40 point clouds of a CAD model are stacked to form a (40, 2048, 3) numpy array, you can easily obtain this data by using following code:

```
import numpy as np
points = np.load("path_of_npy_file")
print(points.shape, type(points))  # (40, 2048, 3), <class 'numpy.ndarray'>
```

Then, you need to put the data into `./dataset/data`, and the contents of directories are as follows:

```
./dataset/data/
├── modelnet40_half1_rm_rotate.txt
├── modelnet40_half2_rm_rotate.txt
├── modelnet_os
│   ├── modelnet_os_test.pickle
│   ├── modelnet_os_train.pickle
│   ├── modelnet_os_val.pickle
│   ├── test [1146 entries exceeds filelimit, not opening dir]
│   ├── train [4194 entries exceeds filelimit, not opening dir]
│   └── val [1002 entries exceeds filelimit, not opening dir]
└── modelnet_ts
    ├── modelnet_ts_test.pickle
    ├── modelnet_ts_train.pickle
    ├── modelnet_ts_val.pickle
    ├── shape_names.txt
    ├── test [1146 entries exceeds filelimit, not opening dir]
    ├── train [4196 entries exceeds filelimit, not opening dir]
    └── val [1002 entries exceeds filelimit, not opening dir]
```

## Training and Evaluation

### Begin training

For ModelNet40 dataset, you can just run:

```
python3 train.py --model_dir=./experiments/experiment_finet/
```

For other dataset, you need to add your own dataset class in `./dataset/data_loader.py`. Training with a lower batch size, such as 16, may obtain worse performance than training with a larger batch size, e.g., 64.

### Begin testing

You need to download the pretrained checkpoint and run:

```
python3 evaluate.py --model_dir=./experiments/experiment_finet --restore_file=./experiments/experiment_finet/test_model_best.pth
```

This model weight is for TS data with Gaussian noise. Note that the performance is a little bit worse than the results reported in our paper (Pytorch implementation).

MegEngine checkpoint for ModelNet40 dataset can be download via [Google Drive](https://drive.google.com/file/d/1nM9bzSYGYA8fsQ0-HSPLo4rOdkG5rxAS/view?usp=sharing).

## Citation

```
@InProceedings{Xu_2022_AAAI,
    author={Xu, Hao and Ye, Nianjin and Liu, Guanghui and Zeng, Bing and Liu, Shuaicheng},
    title={FINet: Dual Branches Feature Interaction for Partial-to-Partial Point Cloud Registration},
    booktitle={Proceedings of the Thirty-Sixth AAAI Conference on Artificial Intelligence},
    year={2022}
}
```

## Acknowledgments

In this project we use (parts of) the official implementations of the following works:

* [RPMNet](https://github.com/yewzijian/RPMNet) (ModelNet40 preprocessing and evaluation)
* [PRNet](https://github.com/WangYueFt/prnet) (ModelNet40 preprocessing)
* [OMNet](https://github.com/megvii-research/OMNet) (Code base)

We thank the respective authors for open sourcing their methods.
