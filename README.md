# This code is for the paper "Remote Sensing Image Ship Detection Based on Dynamic Adjusting Labels Strategy".

Abstract：Ship detection in remote sensing images is vital in military and civil computer vision-related tasks. Nevertheless, the arbitrary orientation and dense arrangement of ship targets impose significant challenges in accurately positioning the bounding boxes. Although research on this problem has progressed, high-precision detection is still limited by angular prediction accuracy. Hence, this paper improves the detection accuracy of arbitrarily oriented and densely arranged ship targets in remote sensing scenes and proposes a Dynamic Adjusting Labels (DAL) strategy based on dense angular coding that is appropriate for angular predictions. Through angle tendency learning guided by dynamically adjusting the ground-truth coding labels in the training process, DAL strengthens the coupling between the angular coding bits and improves the prediction performance of small interval granularity, effectively exploring a dense coding method. Due to the angle interval granularity difference, the learning difficulty of the coding layers and the convergence speed vary significantly. Each layer involves a bit coding loss gradient truncation mechanism to balance the coding layers’ learning strength and enhance the model’s training emphasis on coding bits with small granularity, avoiding the effect of coding bits learning imbalance on angular prediction accuracy. Experimental results on the HRSC2016 ship dataset verify our method’s superiority and competitiveness against current schemes in angular positioning estimation in trials involving accuracy distribution in various intervals, coding bits convergence, and angular learning.

# However, the paper is undergoing review, We will release the source code after the paper is accepted.

# Installation
Please refer to ![install.md](./docs/install.md) for installation and dataset preparation.


# Environments
- Linux support
- python 3.8
- Pytorch 1.7.0+cu101


# Datasets
## datasets
[FGSD-2021] [Baidu Pan](https://pan.baidu.com/s/17L_AGBPu4ux2lUwec0r5JA) [code:dal6]
[HRSC2016] [Baidu Pan](https://pan.baidu.com/s/1guCbReb9ZpsUhMAo04Owbw) [code:dal6]
[DOTA-ship] [Baidu Pan](https://pan.baidu.com/s/1u02CWQfRCJD0VhcPGpBdyA) [code:dal6]

# Datasets structure:
```
├── images
        ├── train
        ├── test
├── labelTxt
        ├── train
        ├── test
├── big_data
        ├── test
```


# Run Details
Please set the code length in 'models/yolov5m.ship.yalm' first.
## Train Process
python train.py  --epochs 200 --batch-size 4 --nbs 16 --device 0
## Test Process
python eval.py
python big_test.py





