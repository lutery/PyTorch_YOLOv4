import os

import numpy as np


def parse_model_cfg(path):
    '''
    解析yolo *.cfg文件
    '''
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    # 处理各种情况下的cfg文件路径，确保能够读到文件
    if not path.endswith('.cfg'):  # add .cfg suffix if omitted
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # add cfg/ prefix if omitted
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    # 去除空行和注释行
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    # modefs是一个列表，里面存放每个模块的定义
    # 每个模块的定义是一个字典，字典的key是参数名，value是参数值
    # 例如，卷积层的定义是一个字典，里面存放卷积层的参数，比如filters, size, stride, pad, activation等
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block [代表一个新的模块]
            mdefs.append({}) # 每个模块的定义是一个字典
            mdefs[-1]['type'] = line[1:-1].rstrip() # 对于新模块，则将其类型添加到type中
            if mdefs[-1]['type'] == 'convolutional': # 如果type时卷积层，则预先添加一些默认值，比如默认不开启bias（后续可以看到使用了bn则不适用bias）
                mdefs[-1]['batch_normalize'] = 0  # pre-populate with zeros (may be overwritten later)
        
        else:
            # 如果不是新模块，则是当前模块的参数
            # 处理当前模块的参数
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                # 处理anchors参数,因为anchors参数是一个字符串，里面存放的是一组坐标值
                # 例如，anchors=10,13, 16,30, 33,23
                # 需要将其转换为numpy数组，形状为(-1, 2)
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                # 处理一些其他可能时用逗号分隔的参数，比如from, layers, mask
                # 例如，from=0,1,2
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                # 处理常规参数，数字转为数字，其余则是字符串
                if val.isnumeric():  # return int or float
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    # 校验参数合法性
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'atoms', 'na', 'nc']

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(u), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (u, path)

    return mdefs


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options
