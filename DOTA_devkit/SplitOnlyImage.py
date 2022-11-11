import os
import numpy as np
import cv2
import copy
import dota_utils as util

class splitbase():
    def __init__(self,
                 srcpath,
                 dstpath,
                 gap=100,
                 subsize=1024,
                 ext='.png'):
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        self.srcpath = srcpath
        self.outpath = dstpath
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.srcpath = srcpath
        self.dstpath = dstpath
        self.ext = ext
    def saveimagepatches(self, img, subimgname, left, up, ext='.png'):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.dstpath, subimgname + ext)
        cv2.imwrite(outdir, subimg)

    def SplitSingle(self, name, rate, extent, num, nums):
        img = cv2.imread(os.path.join(self.srcpath, name + extent))
        assert np.shape(img) != ()

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'

        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]
        
        left, up = 0, 0
        i = 1
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                subimgname = outbasename + str(left) + '___' + str(up)
                self.saveimagepatches(resizeimg, subimgname, left, up)
                print(f'processing {num}/{nums}: {i}')
                i += 1
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def splitdata(self, rate):
        
        imagelist = util.GetFileFromThisRootDir(self.srcpath)
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        nums = len(imagenames)
        for i, name in enumerate(imagenames):
            self.SplitSingle(name, rate, self.ext, i+1, nums)
if __name__ == '__main__':
    # import sys
    # from pathlib import Path
    # FILE = Path(__file__).resolve()
    # ROOT = FILE.parents[0]  # YOLOv5 root directory
    # if str(ROOT) not in sys.path:
    #     sys.path.append(str(ROOT))  # add ROOT to PATH
    # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    
    # img_path = ROOT / 'big_detect/yuantu'
    # split_path = ROOT / 'big_detect/split'
    
    split = splitbase(#img_path,
                      #split_path,
                      '/media/lrs/3D242FF42A23BC10/pcf/yolov5_obb-bsct/big_detect/yuantu',
                      '/media/lrs/3D242FF42A23BC10/pcf/yolov5_obb-bsct/big_detect/split',
                      gap=400,
                      subsize=800,
                      ext='.tif')
    split.splitdata(1)