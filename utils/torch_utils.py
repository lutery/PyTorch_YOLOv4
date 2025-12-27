# PyTorch utils

import logging
import math
import os
import time
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    是一个上下文管理器，用于分布式训练中进程间的同步，确保只有指定的主进程（通常是 local_rank 为 0 的进程）先执行某些关键操作，其它进程则在此处等待
    仅仅只是确认分布式训练时，进程之间的同步

    主进程进来先到yield
    从进程进行进入到local_rank not in [-1, 0]:内部，激活主进程
    从进程到yield则停止
    主进程因为激活开始执行外部的代码
    完成后，主进程执行 if local_rank == 0:的代码激活从进程
    这样就保证了这个期间的代码是主进程先执行的
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # 在 yield 处，进入上下文管理器内部代码块，此时主进程和等待的从进程都可以继续执行上下文内的代码，但从进程是在等待主进程先完成关键操作
    yield
    if local_rank == 0:
        torch.distributed.barrier() 


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        logger.info(f'Using torch {torch.__version__} CPU')

    logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    # torch.cuda.synchronize()：将cuda操作更改有同步方式，这样可以i更加准确的统计显卡的耗时
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def is_parallel(model):
    # 如果模型是并行的，返回True
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, img_size, img_size),), verbose=False)[0] / 1E9 * 2
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.9f GFLOPS' % (flops)  # 640x640 FLOPS
    except (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")

'''
在 PyTorch 里，`nn.Linear(in_features, out_features)` 的权重参数 `weight` 的形状**固定定义**为：

- `weight.shape == (out_features, in_features)`
- `bias.shape == (out_features,)`

原因是线性层的计算通常写成（对单个样本）：

\[
y = x W^T + b
\]

其中：

- \(x\) 形状是 `(in_features,)`（或 batch 情况 `(N, in_features)`）
- \(W\) 形状是 `(out_features, in_features)`，所以 \(W^T\) 是 `(in_features, out_features)`
- 这样 \(x (in\_features) \times W^T (in\_features, out\_features) \Rightarrow y (out\_features)\)

对应到你的代码：

```python
filters = model.fc.weight.shape[1]
```

- `model.fc` 是最后的全连接层
- `model.fc.weight.shape[1]` 就是 `in_features`（输入特征维度）
- `model.fc.weight.shape[0]` 就是 `out_features`（输出类别数，比如 ImageNet 的 1000 类）

所以你看到的现象“`shape[1]` 是输入特征维度、`shape[0]` 是输出特征维度”，不是巧合，而是 **PyTorch 对 Linear 权重张量的约定**。
'''
def load_classifier(name='resnet101', n=2):
    '''
    加载指定的预训练模型，然后更改其最后的全连接层以适应新的分类任务
    
    :param name: Description
    :param n: Description
    '''

    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True) # 加载预训练模型

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1] # 这里shape[1]是输入的特征数量，因为shape[0]是输出的类别数量 todo 为啥？
    # 然后重新创建一个全连接层
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)   # 创建新的偏置
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True) # 创建新的权重
    model.fc.out_features = n # 修改输出类别数量
    return model


def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            gs = 32  # (pixels) grid size
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    '''
    a: ema模型优化器
    b: 模型
    include: 那些需要记录
    exclude: 哪些不需要记录
    '''
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        # 如果有传入include(len(include)) 则判断k是否在include里面
        # 如果k以_开头则不记录，因为是内部属性，理论上不能被外部记录，否则可能会引入一些不合适的值
        # 如果k在待排出列表也不记录
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            # 将属性直接在a中创建并拷贝
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.

    是一个用于实现 指数移动平均（Exponential Moving Average, EMA） 的类。它的主要作用是对模型的参数（权重和缓冲区）进行平滑更新，从而生成一个更稳定的模型版本，通常用于提升模型的泛化能力和性能
    EMA 是一种平滑技术，用于对模型的参数进行加权平均更新。相比于直接使用当前训练的模型参数，EMA 通过引入历史参数的加权平均，能够减少训练过程中参数的波动，生成一个更稳定的模型版本

    有点像强化学习算法中的Target Network

    训练过程中：
    在每个训练步骤后调用 update 方法，更新 EMA 模型的参数。

    验证和测试：
    使用 EMA 模型进行验证和测试，通常可以获得更好的性能。

    保存模型：
    在训练结束时，将 EMA 模型保存为最终的模型版本。
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        # 提取模型的参数，如果是并行模型，则提取module.module_list
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        # 将ema模型的参数设置为不需要梯度更新
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters 这里相当于是软更新，将model的参数按照比例更新到ema模型中
        # todo update_attr是将属性直接覆盖到ema中，而update是将model软更新到ema，这不是冲突吗？ todo 对比两个的调用地方
        with torch.no_grad():
            self.updates += 1
            # 计算权重更新衰减系数
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            # 将model的参数更新到ema模型中
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        # 将当前模型的某些属性（如 model.training）同步到 EMA 模型中。
        # 遍历模型的属性，根据 include 和 exclude 的条件，将属性从当前模型复制到 EMA 模型
        # 查看在哪里调用 在模型优化器调用完 step后，会调用
        copy_attr(self.ema, model, include, exclude)
