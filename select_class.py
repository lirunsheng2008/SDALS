######################用于在数据集中筛选类别###################


# -*- coding: utf-8 -*-
import dota_utils as util
import os
import numpy as np
from PIL import Image
import cv2
import random
import  shutil
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint  # 多边形
import time
import argparse


## trans dota format to  (cls, c_x, c_y, Longest side, short side, angle:[0,179))
def dota2LongSideFormat(imgpath, txtpath, dstpath, extractclassname):
    
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folder
    os.makedirs(dstpath)  # make new output folder
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = util.parse_dota_poly(fullname)
        
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.tif')  # img_fullname='/.../P000?.png'

        # print img_w,img_h
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            num_gt = 0
            for i, obj in enumerate(objects):
                num_gt = num_gt + 1  # 为当前有效gt计数
                poly = obj['poly']  # poly=[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                poly = np.float32(np.array(poly))

                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])  # id=类名的索引 比如'plane'对应id=0
                else:
                    print('预定类别中没有类别:%s;已将该box排除,问题出现在该图片中:%s' % (obj['name'], fullname))
                    num_gt = num_gt - 1
                    continue
                
                if id > 15 or id < 0:
                    print('id problems,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                    c_x, c_y, longside, shortside, theta_longside))
                
                #outline = str(extractclassname[id]) + ' ' + ' '.join(list(map(str, poly)))
                outline = ' '.join(list(map(str, poly))) + ' ' + str(extractclassname[id]) + ' ' + obj['difficult']
                f_out.write(outline + '\n')  # 写入txt文件中并加上换行符号 \n

        if num_gt == 0:
            os.remove(os.path.join(dstpath, name + '.txt'))  #
            os.remove(img_fullname)
            #os.remove(fullname)
            print('%s 图片对应的txt不存在有效目标,已删除对应图片与txt' % img_fullname)
    print('已完成文件夹内DOTA数据形式到长边表示法的转换')



def delete(imgpath, txtpath):
    filelist = util.GetFileFromThisRootDir(txtpath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgpath, name + '.tif')  # img_fullname='/.../P000?.png'
        if not os.path.exists(img_fullname):  # 如果文件bu存在
            os.remove(fullname)
            


if __name__ == '__main__':

    dota2LongSideFormat(r'/media/lrs/3D242FF42A23BC10/HQ/improved_OBB/Cos_DOTA/dataset/newplane_15/images/train',    #大图切割后的图所在位置
                        r'/media/lrs/3D242FF42A23BC10/HQ/improved_OBB/Cos_DOTA/dataset/newplane_15/labelTxt/train', #txt标签路径
                        r'/media/lrs/3D242FF42A23BC10/HQ/improved_OBB/Cos_DOTA/dataset/newplane_15/labelTxt/T',
                        util.new_plane)      #要链接到对应的名字列表




