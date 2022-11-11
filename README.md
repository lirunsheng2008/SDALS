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





