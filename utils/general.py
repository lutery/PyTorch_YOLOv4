# General utils

import glob
import logging
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import yaml

from utils.google_utils import gsutil_getsize
from utils.metrics import fitness, fitness_p, fitness_r, fitness_ap50, fitness_ap, fitness_f   
from utils.torch_utils import init_torch_seeds

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def check_git_status():
    # Suggest 'git pull' if repo is out of date
    if platform.system() in ['Linux', 'Darwin'] and not os.path.isfile('/.dockerenv'):
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def check_dataset(dict):
    # Download dataset if not found locally
    '''
    下载数据集，如果本地不存在则下载，需要在yaml文件中指定下载链接
    '''
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and len(s):  # download script
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    torch.hub.download_url_to_file(s, f)
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))  # unzip
                else:  # bash script
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
            else:
                raise Exception('Dataset not found.')


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def labels_to_class_weights(labels, nc=80):
    '''
    labels: 标签
    nc：定义的分类数

    返回每个样本出现的频率
    '''
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh] 
    # 这行代码是在计算每个类别在训练数据集中出现的次数（频率）
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1 
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    '''
    根据每个样本的标签和类别权重计算图像权重
    labels：数据集中所有图像的标签列表，每个元素是一个图像的标签数组
    nc=80：类别总数，默认80（COCO数据集）
    class_weights=np.ones(80)：每个类别的权重，默认为全1数组
    '''
    # Produces image weights based on class mAPs
    n = len(labels) # 获取数据集中图像的总数量，例如：如果数据集有5000张图像，则 n=5000
    '''
    labels[i][:, 0]：
    获取第i张图像的所有目标的类别索引（标签的第0列）
    例如：[0, 1, 0, 2] 表示这张图有4个目标，类别分别是0,1,0,2

    .astype(np.int)：
    转换为整数类型，确保可以作为 bincount 的输入

    np.bincount(..., minlength=nc)：

    统计每个类别在当前图像中出现的次数
    minlength=nc 确保输出数组长度为 nc
    例如：[2, 1, 1, 0, 0, ...] 表示类别0出现2次，类别1出现1次，类别2出现1次

    列表推导式 + np.array()：
    对每张图像都执行上述统计操作
    最终得到形状为 [n, nc] 的数组
    '''
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # x shape is (n, 4)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, ECIoU=False, eps=1e-9):
    '''
    计算两个边界框之间的IoU（交并比）。box1是一个4维的边界框，box2是一个nx4维的边界框。
    参数说明：
    传参示例：pbox.T, tbox[i], x1y1x2y2=False, CIoU=True
    '''
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    # 这里是根据传递的参数坐标形式计算两个预测框的左上角和右下角的坐标
    # 将坐标统一转换为 左上角和右下角的坐标形式（x1, y1, x2, y2）
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area 计算两个边界框的交集面积。通过计算两个边界框的重叠区域的宽度和高度来得到交集面积。
    # 这里使用了torch.min和torch.max来计算重叠区域的坐标，并使用clamp(0)来确保重叠区域的宽度和高度不为负数。最后通过相乘得到交集面积。
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area 计算并集大小，eps时为了防止除零错误
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # 得到IoU值
    iou = inter / union
    # 这里确认是否需要计算GIoU、DIoU、CIoU、EIoU或ECIoU
    if GIoU or DIoU or CIoU or EIoU or ECIoU:
        # 以下感觉是在寻找两个边界框的最小外接矩形的宽度和高度
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU or EIoU or ECIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # 这是在计算什么？数学原理？
            # 计算最小外接矩形的对角线平方
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            # 这里计算什么？数学原理？
            # 计算两个边界框中心点之间的欧氏距离的平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                # DIoU 是在计算什么？数学原理？
                # IoU: 标准交并比，后面一项是对中心点距离的惩罚
                '''
                考虑中心点距离：即使 IoU 相同，中心点越近，DIoU 越大
                收敛更快：提供了明确的优化方向（让中心点靠近） 
                非重叠情况：即使两个框不重叠，也能提供有意义的梯度
                '''
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                # CIoU 是在计算什么？数学原理？
                # 对比DIou增加了宽高比的一致性度量和权重系数（动态调整）
                # 衡量两个框的宽高比是否一致
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    # 根据 IoU 和 v 动态调整宽高比项的权重
                    # # 当 IoU 很大时（框重叠度高）/ # 权重较大，更关注宽高比
                    # # 当 IoU 很小时（框重叠度低）
                    # # 权重较小，先关注位置和大小
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU: # Efficient IoU https://arxiv.org/abs/2101.08158
                # EIoU 是在计算什么？数学原理？
                rho3 = (w1-w2) **2  # 宽度差的平方
                c3 = cw ** 2 + eps  # 外接矩形宽度的平方
                rho4 = (h1-h2) **2 # 高度差的平方
                c4 = ch ** 2 + eps # 外接矩形高度的平方
                # # CIoU 的问题：
                # 使用 arctan 来衡量宽高比，计算复杂，收敛慢
                # # EIoU 的改进：
                # 直接惩罚宽度差异和高度差异
                return iou - rho2 / c2 - rho3 / c3 - rho4 / c4  # EIoU
            elif ECIoU:
                # ECIoU 是在计算什么？数学原理？
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                rho3 = (w1-w2) **2 # 宽度差的平方
                c3 = cw ** 2 + eps # 外接矩形宽度的平方
                rho4 = (h1-h2) **2 # 高度差的平方
                c4 = ch ** 2 + eps # 外接矩形高度的平方
                return iou - v * alpha - rho2 / c2 - rho3 / c3 - rho4 / c4  # ECIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            # 
            c_area = cw * ch + eps  # convex area 计算外接矩形的大小
            '''
            C: 外接矩形面积
            U: 两个框的并集面积（斜线区域）
            C - U: 灰色区域（外接矩形中的空白部分）

            GIoU惩罚项: (C - U) / C
            - 两个框越接近，C-U 越小，惩罚越小
            - 两个框越分散，C-U 越大，惩罚越大
            '''
            return iou - (c_area - union) / c_area  # GIoU 这里是在计算什么？数学原理？
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    # 这里T将box1和box2的形状从 (N, 4) 和 (M, 4) 转换为 (4, N) 和 (4, M)
    # 然后一次性计算所有边界框的面积
    area1 = box_area(box1.T) # 计算每个边界框的面积，box1.T的形状是4xn， area shape is (n,)
    area2 = box_area(box2.T) # 计算每个边界框的面积，box2.T的形状是4xm， area shape is (m,)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # 2: 获取的是右下角坐标
    # torch.min(box1[:, None, 2:], box2[:, 2:]): 是计算右下角的坐标最小值
    # 反之则torch.max(box1[:, None, :2], box2[:, :2]) 是计算左上角坐标的最大值
    # 可以画图可知 这里是在计算交集区域的宽度和高度
    # .prod(2) 是将宽度和高度相乘，得到交集面积
    # 其他看md文档，最重要的是要计算每一个box1和box2的iou值
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # 最后计算IoU值，也就是交集面积除以并集面积 shape (N,M)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    非极大值抑制

    params prediction: 模型预测的检测结果 注意：prediction的第一个维度是图片的数量batch_size
    conf_thres: 置信度阈值
    iou_thres: IoU阈值
    merge: 是否合并重叠框
    classes: 只保留指定类别的检测结果
    agnostic: 是否进行类别无关的NMS
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes 预测结果的每一行包含 [x, y, w, h, obj_conf, class_scores...]，从第六个元素开始就是预测类别的one-hot编码，所以 类别总数等于列数减去5
    xc = prediction[..., 4] > conf_thres  # candidates 由于预测的结果在第五个位置是目标置信度，所以这里是获取所有目标置信度大于阈值的预测结果

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height 限制边界框的最小和最大宽高 todo 为啥？不是基于原图的吗？
    max_det = 300  # maximum number of detections per image 限制每个图像的最大检测数量
    time_limit = 10.0  # seconds to quit after 设置超时时间，防止处理时间过长
    redundant = True  # require redundant detections 这个是啥意思？ todo
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img) 如果类别数大于1，则表示预测的目标存在多种类别的可能性

    t = time.time() # 计时开始
    output = [torch.zeros(0, 6)] * prediction.shape[0] # output = [torch.zeros(0, 6)] * batch_size 初始化输出列表，每个元素对应一张图像的检测结果，初始为空张量，形状为 (0, 6)，表示没有检测到任何目标 todo 具体的作用
    for xi, x in enumerate(prediction):  # image index, image inference 便利每张图像的预测结果，xi是图像索引，x是该图像的预测结果
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # xc[xi] is (n,) boolean tensor，提取对应索引图片的置信度掩码
        x = x[xc[xi]]  # confidence 将提取出来的掩码应用到预测结果上，保留置信度大于阈值的预测结果

        # If none remain process next image
        if not x.shape[0]: 
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf  将置信度乘以类别概率，得到最终的类别置信度

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4]) # 将yolo格式的边界框转换为xyxy格式

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            # x shape is (n, 5 + num_classes)
            # x[:, 5:] > conf_thres # shape is (n, num_classes)，类型bool，第 k 个框在第 c 类上的置信度是否过阈值
            # .nonzero(as_tuple=False) # shape is (m, 2)，对这个布尔矩阵取非零（True）位置索引，满足阈值的 (框, 类别) 组合数量（注意 m 可能大于 n，因为一个框可能命中多个类别）
            # 每一行是 [row_index, col_index]，也就是：
            # row_index：框的索引（0..n-1）
            # col_index：类别索引（0..nc-1）
            # .T 然后解包给 i, j，T 把 (m, 2) 转成 (2, m)，于是：
            #    i.shape == (m,)：每条命中对应的框索引
            #    j.shape == (m,)：每条命中对应的类别索引
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            # box.shape == (n, 4)，每行是 [x1, y1, x2, y2]
            # box[i].shape == (m, 4)，取出每个命中项对应的框坐标（同一个原始框如果命中多个类，会重复出现多次）
            # x[i, j + 5, None]
            # 把类别索引 j(0..nc-1) 平移到 x 里对应类别置信度所在列（从第 6 列开始）
            # x[i, j + 5] 是 高级索引：逐元素取出每个命中项的那一个类别置信度，x[i, j + 5].shape == (m,)
            # 末尾的 , None 是为了扩维成列向量：x[i, j + 5, None].shape == (m, 1)
            # j[:, None].float()：j.shape == (m,) → j[:, None].shape == (m, 1)，.float()：类别 id 转 float（因为后面整体拼成一个浮点张量）
            # torch.cat(..., 1)：把三块在列维度拼接(m, 4) + (m, 1) + (m, 1) → (m, 6)，最终新的 x：x.shape == (m, 6)列含义：[x1, y1, x2, y2, conf, cls]
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1) # 这句是在把“命中”展开成最终 NMS 需要的检测格式：(x1, y1, x2, y2, conf, cls)。
        else:  # best class only
            # 如果只是单标签，那么x[:, 5:] 提取出类别置信度部分
            # .max(1, keepdim=True) 在类别维度上取最大值，返回值是 (values, indices)
            # values：每个框的最高类别置信度 scores，shape (n, 1)
            # indices：每个框对应的最高类别索引 cls，shape (n, 1)
            conf, j = x[:, 5:].max(1, keepdim=True)
            # conf.view(-1) > conf_thres # shape is (n,) boolean tensor，筛选出置信度过阈值的框
            # torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres] # shape is (m, 6)，仅保留置信度过阈值的框
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            # 类别过滤，有的时候仅需要保留指定类别的检测结果
            # x.shape == (n, 6) 每行：[x1, y1, x2, y2, conf, cls] 其中 x[:, 5] 是类别 id（一般是 float，但值像 0.0, 1.0, ...）
            # x[:, 5:6] 取第 6 列（类别列），但用 5:6 保持二维： shape == (n, 1) # 如果写成 x[:, 5]，shape 会是 (n,)；这里保留 (n,1) 是为了后面更方便广播比较。
            # torch.tensor(classes, device=x.device) 把 Python 的 classes（例如 [0, 2, 7]）变成 tensor，并放到和 x 同一设备（CPU/GPU）上，避免 device mismatch 报错。
            # torch.tensor(classes).shape == (k,)，k 是要保留的类别数量。
            # (x[:, 5:6] == torch.tensor(classes, device=x.device))
            # 左边 (n, 1) ， 右边 (k,) 可视为 (1, k)， 比较结果变为：(n, k) 的布尔矩阵 = 含义：第 i 个框的类别是否等于 classes 里的第 j 个类别。结果 shape == (n, k)，dtype 为 bool
            # .any(1) 在维度 1（类别列表维度 k）上做 “任意为 True”：shape: (n, k) -> (n,) 含义：对每个框来说，只要它的 cls 等于 classes 中任意一个类别，就标记为 True。
            # x[mask]，mask.shape == (n,) 的布尔索引， 过滤后：x.shape == (m, 6)，其中 m <= n，表示保留下来的框数量
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            # 如果没有剩余的检测结果，继续处理下一张图像
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes 实现 “按类分组的 NMS”（class-aware NMS） c.shape == (n, 1)
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores 后面把 c 加到 box 坐标上，会让不同类别的框在坐标空间里被“拉开很远”，从而即使它们物理上重叠很大，NMS 也不会把跨类别的框相互抑制。
        # boxes.shape == (n, 4) ； scores.shape == (n,)
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres) # i.shape == (k,)
        # 按 scores 从高到低排序
        # 依次取最高分框，移除与其 IoU > iou_thres 的其它框
        # 返回保留下来的索引
        if i.shape[0] > max_det:  # limit detections
            # 如果一个图像的检测结果过多，则只保留前 max_det 个最高置信度的结果
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix 过滤出与保留框重叠的其他框
            weights = iou * scores[None]  # box weights 根据 IoU 乘以 置信度计算权重，这里要注意 iou是bool类型，True/False会被转换为1/0
            # 对每个保留框，把与它重叠的一堆框按 score 加权求平均，得到一个“更稳”的框位置
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                # todo 这步的作用是啥？
                i = i[iou.sum(1) > 1]  # require redundancy iou.sum(1)：对每个保留框，看它匹配到多少个重叠框 ； > 1 表示至少有“冗余/重叠”的其它框参与融合（不仅仅是它自己）

        output[xi] = x[i] # 最终输出当前图片的检测结果： x[i].shape == (k_final, 6) 每行仍是 [x1, y1, x2, y2, conf, cls]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def strip_optimizer(f='weights/best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    #x['model'].half()  # to FP16
    #for p in x['model'].parameters():
    #    p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
