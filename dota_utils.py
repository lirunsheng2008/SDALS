# -*- coding: utf-8 -*-
import sys
import codecs
import numpy as np

import shapely.geometry as shgeo
import os
import re
import math
# import polyiou
"""
    some basic functions which are useful for process DOTA data
"""
# For DOTA v1.5
classnames_v1_5 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

plane_tianzhi = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'] #天智杯飞机数据集
new_plane = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',]
#new_plane = ['B-52', 'KC-10', 'E-3', 'B-1B', 'F-16', 'C-130', 'F-15', 'C-17', 'KC-135', 'E-8', 'F-22', 'B-2', 'mordent-737', 'helicopter', 'E-3A', 'others']
ship = ['0', '1', '2', '3']      #HRSC2016数据集
single_plane = ['plane']
PL_ST_RA = ['plane', 'storage-tank', 'roundabout']

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def parse_dota_poly(filename):
    """
        parse the dota ground truth in the format:
        [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    objects = []
    #print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        # count = count + 1
        # if count < 2:
        #     continue
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                    object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                # if splitlines[9] == '1':
                # if (splitlines[9] == 'tr'):
                #     object_struct['difficult'] = '1'
                # else:
                object_struct['difficult'] = splitlines[9]
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [float(splitlines[0]), float(splitlines[1]),
                                     float(splitlines[2]), float(splitlines[3]),
                                     float(splitlines[4]), float(splitlines[5]),
                                     float(splitlines[6]), float(splitlines[7])
                                     ]
            objects.append(object_struct)
        else:
            break
    return objects

def parse_longsideformat(filename):  # filename=??.txt
    """
        parse the longsideformat ground truth in the format:
        objects[i] : [classid, x_c, y_c, longside, shortside, theta]
    """
    objects = []
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    # count = 0
    while True:
        line = f.readline()
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 6) or (len(splitlines) > 6):
                print('labels长度不为6,出现错误,与预定形式不符')
                continue
            object_struct = [int(splitlines[0]), float(splitlines[1]),
                             float(splitlines[2]), float(splitlines[3]),
                             float(splitlines[4]), float(splitlines[5])
                            ]
            objects.append(object_struct)
        else:
            break
    return objects

def parse_dota_poly2(filename):
    """
        parse the dota ground truth in the format:
        [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    objects = parse_dota_poly(filename)
    for obj in objects:
        obj['poly'] = TuplePoly2Poly(obj['poly'])
        obj['poly'] = list(map(int, obj['poly']))
    return objects

def parse_dota_rec(filename):
    """
        parse the dota ground truth in the bounding box format:
        "xmin, ymin, xmax, ymax"
    """
    objects = parse_dota_poly(filename)
    for obj in objects:
        poly = obj['poly']
        bbox = dots4ToRec4(poly)
        obj['bndbox'] = bbox
    return objects
## bounding box transfer for varies format

