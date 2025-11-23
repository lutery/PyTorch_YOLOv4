from utils.google_utils import *
from utils.layers import *
from utils.parse_config import *
from utils import torch_utils

ONNX_EXPORT = False


def create_modules(module_defs, img_size, cfg):
    '''
    param module_defs: list of module definitions,每个模块用一个字典表示
    param img_size: 输入图片的大小
    param cfg: cfg文件的路径 todo 为啥还要传入原始cfg
    大概因为是从yolov3 pytorch版本修改过来的，所以注释中有很多yolov3的注释

    return 构建的层结构，哪些层没有被深层引用False/那些层要被深层引用True
    '''
    # Constructs module list of layer blocks from module configuration in module_defs
    
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels # 处理彩色图片，如果是黑白图片还要手动修改 值是存储每一层的输出通道数，也是下一层的输入通道数的参考
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers 指向更深层的层的列表 用于残差链接，todo 具体如何运作的
    yolo_index = -1

    # 遍历每一个模块的定义
    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            # 处理当前模块是卷积的情况
            # 归一化层的参数
            bn = mdef['batch_normalize']
            # 卷积核的数量
            filters = mdef['filters']
            # 卷积核的尺寸
            k = mdef['size']  # kernel size
            # 处理步长
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                # 如果卷积核是一个尺寸
                # bias=not bn可以看出使用了bn则不使用bias
                # mdef['groups']是否使用分组卷积
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            else:  # multiple-size conv
                # 卷积核是不同的尺寸
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                # 如果使用了bn，则添加bn层
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                # 如果卷积没有使用bn层，则将其的层索引添加到routs中，用于后续的残差链接指定输出到某一层
                routs.append(i)  # detection output (goes into yolo layer)

            # 设置卷积的激活函数
            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'emb':
                modules.add_module('activation', F.normalize())
            elif mdef['activation'] == 'logistic':
                modules.add_module('activation', nn.Sigmoid())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())

        elif mdef['type'] == 'deformableconvolutional':
            # deformableconvolutional 是一种 可变形卷积（Deformable Convolution），其核心思想是通过学习卷积核的偏移量，使卷积操作能够适应目标的形状和大小，从而增强模型对几何变形的建模能力
            # 对于可变形卷积，基本流程和普通卷积类似，就是普通卷积替换为了可变形卷积
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('DeformConv2d', DeformConv2d(output_filters[-1],
                                                       filters,
                                                       kernel_size=k,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       stride=stride,
                                                       bias=not bn,
                                                       modulation=True))
            else:  # multiple-size conv
                modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
                                                          out_ch=filters,
                                                          k=k,
                                                          stride=stride,
                                                          bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())
                
        elif mdef['type'] == 'dropout':
            p = mdef['probability']
            modules = nn.Dropout(p)

        elif mdef['type'] == 'avgpool':
            modules = GAP()

        elif mdef['type'] == 'silence':
            # 占位符，方便后续替换为其他操作，不做任何事情
            filters = output_filters[-1]
            modules = Silence()

        elif mdef['type'] == 'scale_channels':  # nn.Sequential() placeholder for 'shortcut' layer
            # todo 跳过，因为cfg中没有使用
            layers = mdef['from'] 
            filters = output_filters[-1] # 获取上一层的输出通道数
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleChannel(layers=layers)

        elif mdef['type'] == 'sam':  # nn.Sequential() placeholder for 'shortcut' layer
            # todo 跳过 因为cfg中没有使用
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleSpatial(layers=layers)

        elif mdef['type'] == 'BatchNorm2d':
            # 遇到单独的归一化层
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'local_avgpool':
            # 为使用跳过，因为就是构建一个平均池化层
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            avgpool = nn.AvgPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('AvgPool2d', avgpool)
            else:
                modules = avgpool

        elif mdef['type'] == 'upsample':
            # 构建上采样层
            # def['stride']上采样时缩放的倍数
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                # 在使用ONNX导出时，使用指定的大小，应该是ONNX结构不支持吧
                # yolo_index yolo层的索引，也就是最近一次处理的yolo层
                # 用于计算这个yolo层的输入大小
                # 然后计算上采样的大小
                # 由于通道的层数时固定的，所以每个yolo输入的大小都是固定的
                # 即可通过输入的图片尺寸和yolo层的索引来计算
                g = (yolo_index + 1) * 2 / 32  # gain 
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers'] # 是一个数字或者恶意个数字列表
            # 获取指定层的输出通道数的总和
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            # 将指定层的索引添加到routs中
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'route2':  # nn.Sequential() placeholder for 'route' layer
            # todo 跳过 因为cfg中没有使用
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat2(layers=layers)

        elif mdef['type'] == 'route3':  # nn.Sequential() placeholder for 'route' layer
            # todo 跳过 因为cfg中没有使用
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat3(layers=layers)

        elif mdef['type'] == 'route_lhalf':  # nn.Sequential() placeholder for 'route' layer
            # todo 跳过 因为cfg中没有使用
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])//2
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat_l(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            # 短接
            layers = mdef['from'] # 这个应该是要和那个层短接
            filters = output_filters[-1] # 上一层的输出通道数
            # i:表示当前层的索引
            # i + l: 表示找到需要短接的层
            # todo routs：这个的作用是什么
            routs.extend([i + l if l < 0 else l for l in layers])
            # 看起来shortcut中应该还会有一个weights_type的属性
            # 这个属性应该是每个层短接之间的权重把
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # 没有使用，这个应该是yolov3的层
            pass

        elif mdef['type'] == 'reorg':  # yolov3-spp-pan-scale
            # todo 没有使用，但是这里应该就是yolov2中的增加感受野的操作
            filters = 4 * output_filters[-1]
            modules.add_module('Reorg', Reorg())

        elif mdef['type'] == 'yolo':
            # yolo层
            yolo_index += 1 # todo 作用,应该是当前的yolo层的索引吧，因为最初的值是-1
            stride = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7 strides todo 作用
            if any(x in cfg for x in ['yolov4-tiny', 'fpn', 'yolov3']):  # P5, P4, P3 strides
                stride = [32, 16, 8] # 这里应该是针对轻量级检测，减少操作
            # todo 这个操作有点像短接，不过看cfg中好像没有使用到，保存的是需要从哪些层短接吧
            layers = mdef['from'] if 'from' in mdef else []
            # mdef['anchors'][mdef['mask']]：只拿到mask中指定索引的anchors
            # mdef['classes']：分类检测数
            # stride[yolo_index]：todo 只拿一个stride？
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                # 如果定义了from则从其他层拿到偏置，否则就从前一层拿到偏置
                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                # 对前置偏置进行初始化处理，组合要目的在于
                # 这种偏置初始化是基于 YOLOv3 论文 中的 Section 3.3 提出的方法。 todo 查看这个论文的部分
                # 目的是为了：
                # 改善模型在训练初期的稳定性
                # 加快训练收敛速度
                # 优化目标检测的置信度预测
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                #bias[:, 4] += -4.5  # obj
                bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)  # obj (8 objects per 640 image)
                bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
                
                #j = [-2, -5, -8]
                #for sj in j:
                #    bias_ = module_list[sj][0].bias
                #    bias = bias_[:modules.no * 1].view(1, -1)
                #    bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)
                #    bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
                #    module_list[sj][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        elif mdef['type'] == 'jde':
            # todo 暂时跳过，没有这个
            yolo_index += 1
            stride = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7 strides
            if any(x in cfg for x in ['yolov4-tiny', 'fpn', 'yolov3']):  # P5, P4, P3 strides
                stride = [32, 16, 8]
            layers = mdef['from'] if 'from' in mdef else []
            modules = JDELayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                #bias[:, 4] += -4.5  # obj
                bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)  # obj (8 objects per 640 image)
                bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        # 将构建完成的层添加到module_list中
        module_list.append(modules)
        # 存储每一层的输出通道数
        output_filters.append(filters)
        # todo 好像没有看到FPN PAN层？

    # i: 总共有多少层
    # routs_binary: 每一层都是False
    # 如果层数归属于routs，则说明需要被深层引用，可能是残差连接
    # 将被引用的层设置为True，记录再routs_binardy
    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    '''
    yolo层
    '''
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        '''
        param anchors: 根据mask选择的anchors
        param nc: 检测分类数
        param img_size: 检测的输入图片尺寸
        pram yolo_index: 表示当前 YOLO 层在网络中的索引
        param layers: 短接层索引,存储需要从其他层获取输出的层索引
        param stride: 当前yolo层下采样的倍数
        '''
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # 当前 YOLO 层在网络中的索引
        self.layers = layers  #  短接层索引，需要从其他层获取输出的层索引
        self.stride = stride  # layer stride 表示当前 YOLO 层的下采样倍数
        # 短接层的数量
        self.nl = len(layers)  # number of output layers (3)
        # anchors 的数量
        self.na = len(anchors)  # number of anchors (3)
        # 多少个分类
        self.nc = nc  # number of classes (80)
        # 分类数+坐标+置信度
        self.no = nc + 5  # number of outputs (85)
        # nx，ny: 当前yolo层的网格大小，ng是应该是(nx, ny)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        # todo 缩放后的 anchors 为什么要缩放？难道原始的anchors指的是在原始大小中的尺寸？
        self.anchor_vec = self.anchors / self.stride
        # anchors 的宽高张量
        # anchors size: (1, anchor 数量, 1, 1, 2)
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            # 这里应该是针对ONNX的导出支持做准备
            self.training = False
            # img_size[1] // stride：直接得到当前层的输入尺寸
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        '''
        param ng: 应该是当前yolo层的网格大小
        它与输入图像的尺寸和下采样倍数有关： [ ng_w = \frac{\text{input width}}{\text{stride}}, \quad ng_h = \frac{\text{input height}}{\text{stride}} ] 其中，stride 是网络的下采样倍数（如 YOLOv4 中常见的 32、16、8 等）
        它实在创建一个网格坐标，用于后续推理后将每个预测的偏移坐标转换为真实坐标
        '''
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            # torch.arange(self.ny, device=device):生成一个从 0 到 self.ny-1 的一维张量，长度为 self.ny，表示网格在 y 方向的索引
            # torch.arange(self.nx, device=device):生成一个从 0 到 self.nx-1 的一维张量，长度为 self.nx，表示网格在 x 方向的索引
            # torch.meshgrid:用于生成二维网格坐标，输入两个一维张量 [torch.arange(self.ny), torch.arange(self.nx)]，返回两个二维张量
            #  返回两个二维张量：
            #   yv：每一行的值相同，表示网格的 y 坐标。
            #   xv：每一列的值相同，表示网格的 x 坐标。        
            '''
            假设 self.ny = 3，self.nx = 4，则代码执行后：
            yv = [[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [2, 2, 2, 2]]
            
            xv = [[0, 1, 2, 3],
                  [0, 1, 2, 3],
                  [0, 1, 2, 3]]

            yv 表示网格中每个点的 y 坐标
            xv 表示网格中每个点的 x 坐标

            这段代码的作用是 生成网格坐标，用于目标检测任务中将预测框的偏移量加到网格坐标上，从而计算出预测框的实际位置
            在 YOLO 中，输入图像被划分为网格，每个网格负责预测目标框。yv 和 xv 提供了网格的基础坐标，后续会结合预测的偏移量计算目标框的中心点坐标
            '''
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            # torch.stack((xv, yv), 2)：将yv,xv的坐标组合起来，shape为[self.ny, self.nx, 2]，其中每个位置的最后一个维度存储了对应网格点的 (x, y) 坐标
            # .view((1, 1, self.ny, self.nx, 2))：将堆叠后的张量重新调整形状，变为 [1, 1, self.ny, self.nx, 2]
            '''
            self.ny = 3，self.nx = 4。
            xv 和 yv分别为：
            xv = [[0, 1, 2, 3],
                  [0, 1, 2, 3],
                  [0, 1, 2, 3]]
            
            yv = [[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [2, 2, 2, 2]]
            执行 torch.stack((xv, yv), 2) 后
            grid = [[[0, 0], [1, 0], [2, 0], [3, 0]],
                    [[0, 1], [1, 1], [2, 1], [3, 1]],
                    [[0, 2], [1, 2], [2, 2], [3, 2]]]
            执行 .view((1, 1, self.ny, self.nx, 2)) 后
            grid = [[[[[0, 0], [1, 0], [2, 0], [3, 0]],
                    [[0, 1], [1, 1], [2, 1], [3, 1]],
                    [[0, 2], [1, 2], [2, 2], [3, 2]]]]]
            '''
            # 作用生成网格的基础坐标 (x, y)，用于再forward将预测框的偏移量加到网格坐标上，计算实际位置
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            # todo 这边的作用
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        '''
        param p: 当前层的输入特征图， shape：一般为 [batch_size, channels, grid_height, grid_width]。
        形状：
        一般为 [batch_size, channels, grid_height, grid_width]。
        例如，对于一个输入图片大小为 416x416 的网络，假设当前 YOLO 层的下采样倍数为 32，则 grid_height = grid_width = 13。
        channels 的值通常为 anchors × (classes + 5)，其中：
        anchors 是当前 YOLO 层的 anchor 数量（通常为 3）。
        classes 是目标检测的类别数量。
        5 表示每个预测框的 4 个坐标值（x, y, w, h）和 1 个置信度
        param out： 是一个列表，存储了网络中所有中间层的输出
        '''
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            # ASFF 是 Adaptive Spatial Feature Fusion 的缩写，中文翻译为自适应空间特征融合。它是一种用于目标检测的特征融合方法
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]] # 从out中获取当前yolo层的从其他层需要的输入特征图
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13  # bs, channels，输入特征图的尺寸， 输入特征图的尺寸 当前层的输入特征图
            if (self.nx, self.ny) != (nx, ny): # 确保输入特征图的尺寸和记录的尺寸一致
                self.create_grids((nx, ny), p.device)

            # outputs and weights
            # p[:, -n:]：提取特征图的最后n个通道作为权重预测 比如P4层（n=3），会提取最后3个通道：[w_P3, w_P4, w_P5]
            # torch.sigmoid()：将权重压缩到[0,1]范围
            # * (2/n)：权重归一化，确保平均权重约为2/3
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            # [out[self.layers[i]][:, :-n]](http://vscodecontentref/5)：当前层的特征（去掉权重通道）
            # w[:, i:i + 1]：当前层对应的权重
            # 相乘得到当前层的加权特征
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    # [out[self.layers[j]][:, :-n]](http://vscodecontentref/7)：获取第j层的特征（去掉权重通道）
                    # F.interpolate(...)：将其他尺度的特征插值到当前层的尺寸
                    # w[:, j:j + 1]：第j层对应的权重
                    # p +=：累加所有层的加权特征
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            # 如果用于到处，则batch_size只是1
            bs = 1  # batch size
        else:
            # 如果不适用自适应特征融合，那么p的尺寸为
            # batch_size, channels，输入特征图的尺寸， 输入特征图的尺寸
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            # 如果处于训练模式，那么就直接返回了
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            # na=anchors数量
            # nx，ny=输入的特征图尺寸
            # 如果处于到处模式
            m = self.na * self.nx * self.ny # 这里得到的应该是当前yolo层的预测框的数量，也就是anchors数*特征图的大小
            # 可以看作：1. / self.ng： 将网格大小取倒数，用于归一化坐标
            # 其扩展为 (m, 2) 的形状，以便与后续的坐标计算匹配
            # shape 为 (m, 2) 的张量，表示每个网格的宽高，其中2是因为原先ng就是2（nx, ny)
            ng = 1. / self.ng.repeat(m, 1)
            # self.grid 网格的基础坐标 (x, y)，形状为 (1, 1, ny, nx, 2)
            # repeat(1, self.na, 1, 1, 1)表示在索引为1的维度上进行复制扩展，得到shape：1, na, ny, nx, 2)，以匹配 anchor 的数量
            # .view(m, 2)将其展平为 (m, 2)，每个预测框对应一个网格坐标
            # 这个用于将预测的坐标和框高转换为相对全图的比例
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            # self.anchor_wh：状为 (1, na, 1, 1, 2)
            # .repeat(1, 1, self.nx, self.ny, 1)：在索引为2和3的维度上扩展nx和ny ，得到 (1, na, nx, ny, 2)，以匹配特征图的大小
            # view(m, 2)：展平为 (m, 2)，每个预测框对应一个 anchor 的宽高
            # * ng：将 anchor 的宽高归一化到网格尺度，即相对于全图的相对宽高
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng
            # 之所以要进行以上操作，应该是onnx不支持自动广播吧，所以需要手动广播到相同的shape才能进行计算

            # 将输入的特征图p展平为 (m, 85)，其中85是每个预测框的参数数量
            # 这里的参数包括：4个坐标值（x, y, w, h），1个置信度值和nc个分类值
            p = p.view(m, self.no)
            # 得到预测的坐标值
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # 得到预测的宽高值，这里的计算方式不同为了能够兼容onnx模式
            # 为了得到真实的宽高值，那么久需要在onnx推理后进行额外的转换
            # 在实际使用时torch模式和onnx模式在拿到推理结果的后的处理方式会不一致，这点要注意
            # 因为两者的计算方式不同，但是不会造成影响，因为实际训练时，都是拿
            # 中间结果进行计算的，即p
            # 对于两者虽然计算方式不同，但是输入的p都是相同的，就决定了两者最终的预测框结果时一样
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # p[:, 4:5]：预测框的置信度/
            # p[:, 5:self.no]：预测框的类别概率
            # torch.sigmoid(p[:, 4:5])：将置信度压缩到 [0, 1] 范围。
            # torch.sigmoid(p[:, 5:self.no])：将类别概率压缩到 [0, 1] 范围
            # 如果有多个类别，返回 类别概率 × 置信度，如果只有一个类别，则返回 置信度
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            # 返回预测的置信度和类别概率, 将中心点坐标归一化到 [0, 1] 范围（对比原图图的相对位置），预测框的宽高（基于原图尺度的相对宽高）
            return p_cls, xy * ng, wh

        else:  # inference
            # 如果是推理模式，那么先加个p经过sigmoid计算压缩到0~1之间
            # 这里所有的预测值都会做处理，貌似没有对预测框进行筛选匹配，可能在其他地方做了
            io = p.sigmoid()
            # 将预测的坐标当前内部的坐标偏移值加上网格的坐标，得到实际的坐标位置
            io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            # 在推理模式下，这里将预测的检测框尺寸进行转换，得到
            # io[..., 2:4] * 2：将宽高值放大到 [0, 2] 的范围。这是 YOLOv4 中的一种设计，用于增强预测框的表达能力
            # (io[..., 2:4] * 2) ** 2：对宽高值进行平方操作。这是为了确保预测框的宽高值始终为正，同时增加对小目标的敏感性（小值平方后变化更显著
            # * self.anchor_wh：将预测框的宽高与对应的 anchor 宽高相乘，得到实际的宽高值。
            # self.anchor_wh 是一个张量，存储了当前 YOLO 层的 anchor 宽高，已经根据特征图的下采样倍数进行了缩放。
            # 通过以下计算得到实际的宽高值（还是需要结合下采样倍数进行还原，即乘以stride）
            # todo 还是需要实际的调试感受
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            # 这一步是将预测框的坐标和宽高值进行缩放，得到实际的坐标和宽高值,即将预测的结果恢复回相对原图的尺寸，根据下采样的倍率
            # 这里的 stride 是当前 YOLO 层的下采样倍数
            io[..., :4] *= self.stride
            #io = p.clone()  # inference output
            #io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            #io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            #io[..., :4] *= self.stride
            #torch.sigmoid_(io[..., 4:])
            # 展平，使得size=（batch_size, -1, classes+xywh+1)
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class JDELayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride):
        super(JDELayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            i, n = self.index, self.nl  # index in layers, number of layers
            p = out[self.layers[i]]
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

            # outputs and weights
            # w = F.softmax(p[:, -n:], 1)  # normalized weights
            w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # weighted ASFF sum
            p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            for j in range(n):
                if j != i:
                    p += w[:, j:j + 1] * \
                         F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            #io = p.sigmoid()
            #io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            #io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            #io[..., :4] *= self.stride
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) * 2. - 0.5 + self.grid  # xy
            io[..., 2:4] = (torch.sigmoid(io[..., 2:4]) * 2) ** 2 * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            io[..., 4:] = F.softmax(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        '''
        构建yolov4的模型，使用的是Darknet的cfg文件形式解析构建
        '''
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        # todo version seen的作用未知
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        # 打印模型结构，每一层的参数量和计算量，todo 后续可以引入到我的代码中
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, augment=False, verbose=False):
        '''
        x: 图片的tensor数据 batch_size, 3, height, width
        augment: 是否进行增强 
        verbose: 是否打印每一层的输出

        return 返回预测后的检测框，坐标尺寸都是相对于原图
        '''

        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = [] # 存储增强后预测的yolo层的输出结果
            # 下面的操作就是对图片进行增强操作，有缩放，翻转
            # 这里的x仅一个批次，没有多个批次，所以这里仅回for 3次，所以y才只有3个
            for i, xi in enumerate((x, # 不进行处理
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                # self.forward_once(xi)[0]：因为返回的是一个元组，所以这里仅获取预测的结果
                '''
                # - batch_size: 批次大小（如果没有在 forward_once 内部增强，则为原始 batch_size）
                # - total_predictions: 所有 YOLO 层的预测框总数
                #   例如: P3(52×52×3) + P4(26×26×3) + P5(13×13×3) = 8112 + 2028 + 507 = 10647
                # - 85: [x, y, w, h, obj, cls0, cls1, ..., cls79]
                '''
                y.append(self.forward_once(xi)[0])

            # 因为预测的是缩放、翻转的图片，所以这里需要将预测的结果重新在翻转、缩放回来
            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
        '''
        x: 图片的tensor数据 batch_size, 3, height, width
        augment: 是否进行增强 
        verbose: 是否打印每一层的输出
        '''

        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], [] # yolo_out存储yolo层的输出，out存储每一层的输出
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale flip(3)表示具体含义是沿着第3个维度（索引为3）进行翻转
                           torch_utils.scale_img(x, s[1]),  # scale 单纯的缩放图片
                           ), 0)

        for i, module in enumerate(self.module_list):
            # 遍历每一个module
            name = module.__class__.__name__
            #print(name)
            if name in ['WeightedFeatureFusion', 'FeatureConcat', 'FeatureConcat2', 'FeatureConcat3', 'FeatureConcat_l', 'ScaleChannel', 'ScaleSpatial']:  # sum, concat
                # 因为这些层需要从其他层获取输出，所以需要独立的处理
                if verbose:
                    l = [i - 1] + module.layers  # layers 记录层索引：当前层的前一层索引 + 需要融合的层索引 例如：如果当前是第10层，需要融合第5层和第8层，则 l = [9, 5, 8]
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes 打印当前层的输入shape和待融合的输入shape作为对比
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat() 进行处理
            elif name == 'YOLOLayer':
                # 如果是训练模式，则输出的是batch_size, channels(255), grid_h, grid_w
                # 如果是推理模式，则输出的是batch_size, num_anchors*grid_h*grid_w, 85
                yolo_out.append(module(x, out))
            elif name == 'JDELayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                #print(module)
                #print(x.shape)
                x = module(x)

            # self.routs表示那些层的输出会被其他层引用，所以这里是判断，如果是需要被其他层引用，则保存当前层的输出
            out.append(x if self.routs[i] else [])
            if verbose:
                # 调试信息，打印当前层的输出x的shape
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train 所有层遍历完成了，如果是训练模式则直接返回yolo的输出即可
            return yolo_out
        elif ONNX_EXPORT:  # export
            # ONNX模式下，yolo_out是一个列表，里面存储了每个yolo层的输出    
            # 格式是：分类置信度，xy,wh
            # torch.cat(x, 0)是将每个yolo层的输出进行拼接，得到总分类置信度，xy,wh
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # x[0]代表分类置信度
            # x[1:3]代表xy,wh，通过cat使其拼接起来
            # 最终得到scores（置信度）, boxes（坐标，但还是相对于原图得到比例）: 3780x80, 3780x4
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test 如果是对于pytorch的推理模式
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs 将所有yolo层的预测框进行拼接
            if augment:  # de-augment results
                # 如果开启了图片增强，那么对于输入的特征图是有做缩放、镜像处理，所以需要进行逆处理
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1) # 再次拼接回来
            return x, p # 返回所有的预测框和每个预测框对应物体类别的的置信度

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    '''
    获取yolo层的索引，识别方法是根据名字带有yolo关键字的识别
    '''
    return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ in ['YOLOLayer', 'JDELayer']]  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights', saveto='converted.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)
    ckpt = torch.load(weights)  # load checkpoint
    try:
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
        save_weights(model, path=saveto, cutoff=-1)
    except KeyError as e:
        print(e)

def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip()
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if len(weights) > 0 and not os.path.isfile(weights):
        d = {''}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
