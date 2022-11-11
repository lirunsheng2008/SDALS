# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly
import numpy as np

from utils.evaluation_utils import mergebypoly, draw_DOTA_image, evaluation_trans, image2txt
from utils import polyiou
from statistics import mean

from utils.datasets import create_dataloader

def poly2txt(poly, classname, conf, img_name, out_path, pi_format=False):
    """
    将分割图片的目标信息填入原始图片.txt中
    @param robx: rbox:[tensor(x),tensor(y),tensor(l),tensor(s),tensor(θ)]
    @param classname: string
    @param conf: string
    @param img_name: string
    @param path: 文件夹路径 str
    @param pi_format: θ是否为pi且 θ ∈ [-pi/2,pi/2)  False说明 θ∈[0,179]
    """
    poly = np.float32(torch.tensor(poly).cpu())
    poly = np.int0(poly).reshape(8)

    splitname = img_name.split('__')
    oriname = splitname[0]

    lines = img_name + ' ' + conf + ' ' + ' '.join(list(map(str, poly))) + ' ' + classname
    if not os.path.exists(out_path):
        os.makedirs(out_path)  # make new output folder
    with open(str(out_path + '/' + oriname) + '.txt', 'a') as f:
        f.writelines(lines + '\n')

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        len_encode=7
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project), exist_ok=exist_ok, mkdir=True)  # increment run
    if save_img:
        save_img_dir = increment_path(save_dir /'results', mkdir=True)  # im.jpg

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        # pred: list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det, len_encode=len_encode)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            if save_img:
                save_path = str(save_img_dir / p.name)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale polys from img_size to im0 size
                pred_poly = scale_polys(im.shape[2:], pred_poly, im0.shape)
                det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *poly, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        label = '%s %.2f' % (names[int(cls)], conf)
                        classname = '%s' % names[int(cls)]
                        conf_str = '%.3f' % conf
                        poly2txt(poly, classname, conf_str, p.stem, str(save_dir / 'result_txt/result_before_merge'))
                                              
                    if save_img or save_crop or view_img:  # Add poly to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.poly_label(poly, label, color=colors(c, True))
                        if save_crop: # Yolov5-obb doesn't support it yet
                            pass

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def parse_opt():
    model_yalm = ROOT / 'models/yolov5m.hrsc.yaml'
    if isinstance(model_yalm , dict):
        yolo_yaml = model_yalm   # model dict
    else:  # is *.yaml
        import yaml  # for torch hub
        with open(model_yalm , encoding='ascii', errors='ignore') as f:
            yolo_yaml = yaml.safe_load(f)  # model dict
    len_encode = yolo_yaml['encode']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/fgsd2021_7.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='./dataset/fgsd2021/images/test/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--project', default=ROOT / 'big_test/output', help='save results to project/name')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1024], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default=False, action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS') # 大图验证时将所有类别一起nms
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--len-encode', type=int, default=len_encode, help='length of the angle encode')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                object_struct['difficult'] = 0
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([1.0], prec, [0.0]))
        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(annopath+'/'+imagename+'.txt')

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    #print('check fp:', fp)
    #print('check tp', tp)


    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def evaluation(detoutput, imageset, annopath, classnames, use_07_metric=False):
    """
    评估程序
    @param detoutput: detect.py函数的结果存放输出路径
    @param imageset: # val DOTA原图数据集图像路径
    @param annopath: 'r/.../{:s}.txt'  原始val测试集的DOTAlabels路径
    @param classnames: 测试集中的目标种类
    """
    result_merged_path = str(detoutput + '/result_txt/result_merged')
    result_classname_path = str(detoutput + '/result_txt/result_classname')
    imageset_name_file_path = str(detoutput + '/result_txt')

    evaluation_trans(
        result_merged_path,
        result_classname_path
    )
    print('检测结果已按照类别分类')
    image2txt(
        imageset,
        imageset_name_file_path)
    print('校验数据集名称文件已生成')

    detpath = str(result_classname_path + '/Task1_{:s}.txt')  # 'r/.../Task1_{:s}.txt'  存放各类别结果文件txt的路径
    imagesetfile = str(imageset_name_file_path +'/imgnamefile.txt')  # 'r/.../imgnamefile.txt'  测试集图片名称txt

    iou_thr = np.arange(0.5, 1, 0.05)
    classaps = [[0 for col in range(len(classnames))] for row in range(len(iou_thr))]
    mAP = []
    P = []
    R = []
    for i, iou in enumerate(iou_thr):
        map = 0
        for j, classname in enumerate(classnames):
            detfile = detpath.format(classname)
            if not (os.path.exists(detfile)):
                # print('This class is not be detected in your dataset: {:s}'.format(classname))
                continue
            rec, prec, ap = voc_eval(detpath,
                annopath,
                imagesetfile,
                classname,
                ovthresh=iou,
                use_07_metric=use_07_metric)
            map = map + ap
            # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
            # print('ap: ', ap)
            classaps[i][j] = ap

            if i==0:
                P.append(prec)
                R.append(rec)

        map = map/len(classnames)
        mAP.append(map)
        
    classaps = np.array(classaps)
    
    # # HRSC2016
    # pf = '%20s' + '%11s' * 6
    # LOGGER.info(pf % ('Class','mAP@.5', 'mAP@.65', 'mAP@.75', 'mAP@.85', 'mAP@.95', 'mAP@.5:.95'))
    # pf = '%20s' + '%11.3g' * 6  # print format
    # LOGGER.info(pf % ('all', mAP[0], mAP[3], mAP[5], mAP[7], mAP[9], mean(mAP)))
    # # Print results per class
    # for i, classname in enumerate(classnames):
    #     LOGGER.info(pf % (classname, classaps[0,i], classaps[3,i], classaps[5,i], classaps[7,i], classaps[9,i], np.mean(classaps[:,i])))
    
    # # dota-ship
    # pf = '%20s' + '%11s' * 4
    # LOGGER.info(pf % ('Class','mAP@.5', 'mAP@.60', 'mAP@.70', 'mAP@.80'))
    # pf = '%20s' + '%11.3g' * 4  # print format
    # LOGGER.info(pf % ('all', mAP[0], mAP[2], mAP[4], mAP[6]))
    # # Print results per class
    # for i, classname in enumerate(classnames):
    #     LOGGER.info(pf % (classname, classaps[0,i], classaps[2,i], classaps[4,i], classaps[6,i]))

    # fgsd2021
    pf = '%20s' + '%11s' * 4
    LOGGER.info(pf % ('Class','mAP@.5', 'mAP@.60', 'mAP@.70', 'mAP@.80'))
    pf = '%20s' + '%11.4g' * 4  # print format
    LOGGER.info(pf % ('all', mAP[0], mAP[2], mAP[4], mAP[6]))
    # Print results per class
    for i, classname in enumerate(classnames):
        LOGGER.info(pf % (classname, classaps[0,i], classaps[2,i], classaps[4,i], classaps[6,i]))


def delta_angle(detpath, annopath, imagesetfile, classname, ovthresh=0.5, len_encode=7):
    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(annopath+'/'+imagename+'.txt')

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    
    nd = len(image_ids)
    delta_code = 0
    delta_ang = 0 
    contribution_ang = 0
    n = 0

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    R['det'][jmax] = 1

                    rbox_gt = poly2rbox(polys=R['bbox'][jmax][np.newaxis, :], use_pi=False)
                    rbox_pred = poly2rbox(polys=bb[np.newaxis, :], use_pi=False)

                    ang_gt = np.squeeze(rbox_gt)[4]
                    ang_pred = np.squeeze(rbox_pred)[4]
                    
                    encode_gt = np.zeros(len_encode)
                    encode_pred = np.zeros(len_encode)
                    a_gt = 90.
                    a_pred = 90.
                    
                    for i in range(len_encode):
                        encode_gt[i] = np.sign(ang_gt - a_gt)
                        a_gt += 90 * pow(0.5, i+1) * encode_gt[i]

                        encode_pred[i] = np.sign(ang_pred - a_pred)
                        a_pred += 90 * pow(0.5, i+1) * encode_pred[i]
                        
                    encode_gt[encode_gt==-1] = 0
                    encode_pred[encode_pred==-1] = 0

                    # bit error rate
                    delta_c = abs(encode_gt - encode_pred)
                    delta_code += delta_c

                    # average angle error
                    delta_a = min(abs(ang_gt-ang_pred), 180-abs(ang_gt-ang_pred))
                    delta_ang += delta_a

                    # angle contribution
                    contribution_a = 90 / (delta_a+1e-2) - 1
                    contribution_ang += contribution_a
                    
                    n += 1

    return delta_code, delta_ang, contribution_ang, n

def calculate_delta_ang(detoutput,annopath, classnames, len_encode=7):
    """
    计算平均角度误差和角度贡献率
    @param detoutput: detect.py函数的结果存放输出路径
    @param imageset: # val DOTA原图数据集图像路径
    @param annopath: 'r/.../{:s}.txt'  原始val测试集的labels路径
    @param classnames: 测试集中的目标种类
    """
    
    result_classname_path = str(detoutput + '/result_txt/result_classname')
    imageset_name_file_path = str(detoutput + '/result_txt')
    detpath = str(result_classname_path + '/Task1_{:s}.txt')  # 'r/.../Task1_{:s}.txt'  存放各类别结果文件txt的路径
    
    imagesetfile = str(imageset_name_file_path +'/imgnamefile.txt')  # 'r/.../imgnamefile.txt'  测试集图片名称txt

    # 初始化 
    delta_codes = 0 
    delta_angs = 0
    contribution_angs = 0
    num = 0

    for classname in classnames:
        # print('classname:', classname)
        detfile = detpath.format(classname)
        if not (os.path.exists(detfile)):
            continue

        delta_code, delta_ang, contribution_ang, n = delta_angle(detpath,
                                                                 annopath,
                                                                 imagesetfile,
                                                                 classname,
                                                                 ovthresh=0.5,
                                                                 len_encode=len_encode)
        # bit error rate
        delta_codes += delta_code

        # average angle error
        delta_angs += delta_ang

        # angle contribution
        contribution_angs += contribution_ang

        num += n

    print(f'bit error rate:{delta_codes / num}')
    print(f'average angle error:{delta_angs / num}')
    print(f'angle contribution:{contribution_angs / num}')


if __name__ == "__main__":
    
    # fgsd2021
    classnames = ['Air', 'Was', 'Tar', 'Oth', 'Aus', 'Whi', 'San', 'New', 'Tic', 'Bur', 
                  'Per', 'Lew', 'Sup', 'Kai', 'Hop', 'Mer', 'Fre', 'Ind', 'Ave', 'Sub']  
    # dota-ship
    # classnames = ['ship']
    
    opt = parse_opt()
    
    # detect
    main(opt)
    
    # merge
    detoutput='./big_test/output' 
    result_before_merge_path = str(detoutput + '/result_txt/result_before_merge')
    result_merged_path = str(detoutput + '/result_txt/result_merged')
    mergebypoly(
        result_before_merge_path,
        result_merged_path)
    print('The results have been merged')

    draw_DOTA_image(imgsrcpath='./dataset/fgsd2021/big_data/test/images',
                    imglabelspath='./big_test/output/result_txt/result_merged',
                    dstpath='./big_test/merge',
                    extractclassname=classnames,
                    thickness=opt.line_thickness,
                    extend='.jpg',
                    colors=colors,
                    hide_labels=False)
    
    # mAP
    evaluation(
        detoutput='./big_test/output',
        imageset='./dataset/fgsd2021/big_data/test/images',
        annopath='./dataset/fgsd2021/big_data/test/labelTxt',
        classnames=classnames,
        use_07_metric=True
        )
    
    # verage angle error & angle contribution
    calculate_delta_ang(
        detoutput='./big_test/output',
        annopath='./dataset/fgsd2021/big_data/test/labelTxt',
        classnames=classnames,
        len_encode=opt.len_encode
    )
    

