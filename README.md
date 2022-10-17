# PFRRNet

## Installation
First you have to make sure that you have all dependencies in place.

You can create an anaconda environment called `pfrrnet` using
```
conda env create -n pfrrnet python=3.6 ## recommended python=3.6+
conda activate pfrrnet
sudo pip3 install torch torchvision 
sudo pip3 install numpy scipy matplotlib
sudo pip3 install dlib
sudo pip3 install opencv-python
sudo pip3 install cython
```
Next, compile the extension modules.
```
cd utils/cython
python3 setup.py build_ext -i
```

## Generation
To generate results using a trained model, use
```
python3 main.py -f samples/test.jpg 
```
In addition, to quickly generate results using other SOTA methods for you, we add main_{SOTA_method_names} to generate the results:
```
python3 main_{other method}.py -f samples/test.jpg 
```
Note that we suggest you choose normal image due to dlib restrictions on face capture

## Pretrained Models
In order to facilitate everyone to reproduce all the comparison algorithms in this article at one time, we uniformly package their pre-trained models to Google Drive [models](https://drive.google.com/drive/folders/1lyJhEN2WDlicUH2vmOoug6vykacNqfEq?usp=sharing)

## Evaluation
To eval our PFRRNet , use
```
python benchmark.py
```
```
python benchmark_3D.py
```
<br>
Note that our GPU is Nvidia RTX 3090, and the test environment is cuda V11.1, Pytorch 1.7.

## Training
To train our PFRRNet with wpdc and wing Loss, use
```
## Data
| Data | Download Link | Description |
|:-:|:-:|:-:|:-:|:-:|
| train.configs | [BaiduYun](https://pan.baidu.com/s/1ozZVs26-xE49sF7nystrKQ) or [Google Drive](https://drive.google.com/open?id=1dzwQNZNMppFVShLYoLEfU3EOj3tCeXOD), 217M | The directory containing 3DMM params and filelists of training dataset |
| train_aug_120x120.zip | [BaiduYun](https://pan.baidu.com/s/19QNGst2E1pRKL7Dtx_L1MA) or [Google Drive](https://drive.google.com/open?id=17LfvBZFAeXt0ACPnVckfdrLTMHUpIQqE), 2.15G | The cropped images of augmentation training dataset |
| test.data.zip | [BaiduYun](https://pan.baidu.com/s/1DTVGCG5k0jjjhOc8GcSLOw) or [Google Drive](https://drive.google.com/file/d/1r_ciJ1M0BSRTwndIBt42GlPFRv6CvvEP/view?usp=sharing), 151M | The cropped images of AFLW and ALFW-2000-3D testset |
| LAG Dataset  | [BaiduYun](https://pan.baidu.com/s/1DTVGCG5k0jjjhOc8GcSLOw) or [Google Drive](https://drive.google.com/file/d/1r_ciJ1M0BSRTwndIBt42GlPFRv6CvvEP/view?usp=sharing), | The cropped images of AFLW and ALFW-2000-3D testset |
| Casual Photos | [BaiduYun](https://pan.baidu.com/s/1DTVGCG5k0jjjhOc8GcSLOw) or [Google Drive](https://drive.google.com/file/d/1r_ciJ1M0BSRTwndIBt42GlPFRv6CvvEP/view?usp=sharing),  | The cropped images of AFLW and ALFW-2000-3D testset |

cd training
bash train_pfrrnet.sh
```
## Quantitative Results
 NME2D   | AFLW2000-3D Dataset (68 pts)  | AFLW Dataset (21 pts)
:-: | :-: | :-: 
Method |[0,30],[30,60],[60,90], Mean, Std  | [0,30],[30,60],[60,90], Mean, Std
CDM | -, -, -, -, - | 8.150, 13.020, 16.170, 12.440, 4.040 
RCPR | 4.260, 5.960, 13.180, 7.800, 4.740 | 5.430, 6.580, 11.530, 7.850, 3.240
ESR | 4.600, 6.700, 12.670, 7.990, 4.190 | 5.660, 7.120, 11.940, 8.240, 3.290
SDM | 3.670, 4.940, 9.760, 6.120, 3.210 | 4.750, 5.550, 9.340, 6.550, 2.450 
DEFA  | 4.500, 5.560, 7.330, 5.803, 1.169 | -, -, -, -, - 
3DDFA(CVPR2016)  | 3.780, 4.540, 7.930, 5.420, 2.210 | 5.000, 5.060, 6.740, 5.600, 0.990
Nonlinear(CVPR2018)   | -, -, -, 4.700, - | -, -, -, -, -
DAMDNet(ICCVW19)  | 2.907, 3.830, 4.953, 3.897, 0.837 | 4.359, 5.209, 6.028, 5.199, 0.682 
MFIRRN(iCASSP2021)  | 2.841, 3.572, 4.561, 3.658, 0.705 | 4.321, 5.051, 5.958, 5.110, 0.670 
RADANet(FG2021)  |2.792, 3.583, 4.495, 3.623, 0.696 | 4.129, 4.888, 5.495, 4.837, 0.559
2DAL(T-MM) | 2.750, 3.460, 4.450, 3.550| -, -, -, -, -
PFRRNet (Ours)| 2.616, 3.381, 4.342, 3.446, 0.706| 3.976, 4.578, 5.237, 4.597, 0.515

## Qualitative Results of Dense Aligment and Qualitative Results of 3D Reconstruction 
<img src="https://github.com/Ray-tju/PFRRNet/blob/main/display/qs_small.jpg" width="300" height="300"><img src="https://github.com/Ray-tju/PFRRNet/blob/main/display/qs_3d2_samll.jpg" width="300" height="300">

# Futher Information
If you have any problems with the code, please list the problems you encountered in the issue area, and I will reply you soon.
Thanks for baseline work [3DDFA](https://github.com/cleardusk/3DDFA).
