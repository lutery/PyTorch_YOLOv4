import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
#from models.yolo import Model
from models.models import *
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, fitness_p, fitness_r, fitness_ap50, fitness_ap, fitness_f, strip_optimizer, get_latest_run,\
    check_dataset, check_file, check_git_status, check_img_size, print_mutation, set_logging
from utils.google_utils import attempt_download
from utils.loss import compute_loss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first

logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def train(hyp, opt, device, tb_writer=None, wandb=None):
    '''
    训练
    param hyp: 训练的超参数
    param opt: 训练的参数
    param device: 训练的设备
    param tb_writer: tensorboard的writer
    param wandb: wandb的writer
    '''
    logger.info(f'Hyperparameters {hyp}')
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    # 这里是将本次训练的参数保存到文件中，保持训练的一致性
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    # 加载数据集信息
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    with torch_distributed_zero_first(rank):
        # 这边就是在确保只有一个线程才会执行下载数据集，通过yield操作
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    # single_cls表示只有一个类别，如果不是则从yaml中的获取类别数和类别名称
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    # 如果传入的权重，则加载权重
    # 如果不想下载权重，则传入空字符串
    # 而这里的代码必须要传入空串，因为内部的下载代码是去下载另一个yolov4 scaled的权重
    # 并且下载的路径只能时他们的路径，而不能时自己的
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            # 又是在确保只有一个线程才会执行下载数据集，通过yield操作
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Darknet(opt.cfg).to(device)  # create
        state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(state_dict, strict=False)
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Darknet(opt.cfg).to(device) # create

    # Optimizer
    nbs = 64  # nominal batch size 立项状态的batch_size大小
    #  total_batch_size：实际训练的批次大小，受限于显存大小
    '''
    这行代码的目的是：如果实际批量小于标称批量，则通过多步累积梯度，使得每次优化器更新的等效批量接近 nbs。
    例如：实际 batch size=16，nbs=64，则 accumulate=4，表示每4个batch累积一次梯度再更新参数
    '''
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    '''
    权重衰减缩放（weight_decay）

为了保证不同 batch size 下训练效果一致，需要按等效 batch size 对权重衰减（L2正则）进行缩放。
这行代码根据实际 batch size 和累积步数调整 hyp['weight_decay']，使其与标称 batch size 下的效果一致。
todo 了解数学原理
    '''
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    # todo 了解pg0 pg1 pg2的作用
    # pg0存储的是非偏置、卷积、m\w的权重
    # pg1存储卷积、线性层的权重
    # pg2存储偏置的权重
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'Conv2d.weight' in k:
            pg1.append(v)  # apply weight_decay
        elif 'm.weight' in k:
            pg1.append(v)  # apply weight_decay
        elif 'w.weight' in k:
            pg1.append(v)  # apply weight_decay
        else:
            pg0.append(v)  # all else

    # 选择合适的梯度优化器
    # 这里进队pg0中的参数进行优化，其余的可能要按照accumulate进行手动参数优化
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # add_param_group: 用于向已有的优化器中动态添加新的参数组
    '''
    作用
    允许你为不同的参数组设置不同的超参数（如学习率、权重衰减等）。
    常用于对模型的不同部分（如权重、偏置）采用不同的优化策略
    '''
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    # todo 了解这篇论文
    '''
    YOLOv4 采用如下方式的学习率调度器（cosine annealing）：
### 原因与优势

1. **平滑衰减，避免震荡**  
   余弦退火（cosine annealing）让学习率从初始值平滑地下降到较低值，避免了阶梯式下降带来的训练震荡，有助于模型更稳定地收敛。

2. **前期探索，后期收敛**  
   前期保持较高学习率，有利于模型跳出局部最优、充分探索参数空间；后期逐步减小学习率，有助于模型在最优点附近精细收敛。

3. **提升最终精度**  
   余弦调度器能有效提升目标检测模型的最终精度，已被大量实验和论文验证（如 YOLOv4、YOLOv5、ResNet 等）。

4. **无需手动调整**  
   自动根据 epoch 变化调整学习率，减少了手动设置学习率衰减时机的麻烦。

5. **兼容 warmup**  
   余弦调度器常与 warmup（预热）结合，进一步提升训练初期的稳定性和收敛速度。

### 总结

YOLOv4 采用余弦退火学习率调度器，是为了让训练过程更平滑、收敛更稳定、最终精度更高，同时简化超参数调整。这是现代深度学习目标检测任务中非常主流且有效的做法。
    '''
    # 输入的x是当前是第几个epoch，返回的是当前epoch的学习率
    # todo 绘制曲线
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Logging
    # 可视化配置
    if wandb and wandb.run is None:
        opt.hyp = hyp  # add hyperparameters
        wandb_run = wandb.init(config=opt, resume="allow",
                               project='YOLOv4' if opt.project == 'runs/train' else Path(opt.project).stem,
                               name=save_dir.stem,
                               id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    best_fitness_p, best_fitness_r, best_fitness_ap50, best_fitness_ap, best_fitness_f = 0.0, 0.0, 0.0, 0.0, 0.0
    if pretrained:
        # 加载与训练模型以及超参数
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
            best_fitness_p = ckpt['best_fitness_p']
            best_fitness_r = ckpt['best_fitness_r']
            best_fitness_ap50 = ckpt['best_fitness_ap50']
            best_fitness_ap = ckpt['best_fitness_ap']
            best_fitness_f = ckpt['best_fitness_f']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        # 如果是继续训练，则不能加载预训练模型
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        # 这里表示预训练模型存在问题，需要重新调整训练总轮数
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = 64 #int(max(model.stride))  # grid size (max stride)
    # 这边确认传入的imgsz是64的倍数，opt.img_size是一个list，包含训练和测试的image size
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode 是否进行多GPU训练
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    # 将模型中的所有 BatchNorm 层（如 BatchNorm1d、BatchNorm2d 等）转换为 SyncBatchNorm 层
    # 在分布式数据并行（DDP）训练中，每个 GPU 都会处理一部分数据（即一个 mini-batch）。普通的 BatchNorm 层只会在当前 GPU 的 mini-batch 上计算均值和方差，这可能导致不同 GPU 上的统计信息不一致，进而影响模型的收敛和性
    if opt.sync_bn and cuda and rank != -1:
        # 仅适用于 DDP 模式，普通 DataParallel 模式下使用，可能会报错
        # SyncBatchNorm 需要在每次前向传播时进行跨 GPU 的通信，这会增加一定的通信开销，可能稍微降低训练速度。
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # EMA 创建一个EMA模型，用于后续的模型评估和保存
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode 是否开启DDP分布式训练模式
    # DDP分布式训练：每个GPU具备完整的模型，每个GPU分配一部分数据进行训练，训练完成后通过通信机制将每个GPU的梯度进行平均更新模型参数
    if cuda and rank != -1:
        # 这个模式是由pytorch提供的 
        # 参数：  
        # model：要并行化的模型
        # device_ids：指定要使用的GPU设备ID列表
        # output_device：指定输出设备ID
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class 获取label的最大值，也就是最大的分类数
    nb = len(dataloader)  # number of batches 有多少个batch
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)
    '''
    mlc (max label class): 是数据集中出现的最大类别索引
    nc (number of classes): 是配置文件中定义的类别总数
    这个断言的目的是：

    验证数据集标签的有效性：确保所有标签的类别索引都在有效范围内
    防止索引越界：标签索引必须在 [0, nc-1] 范围内
    检查数据集一致性：确保数据集标签与配置文件中定义的类别数相匹配
    '''

    # Process 0
    if rank in [-1, 0]:
        # 确保测试集合仅在主进程中创建
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader = create_dataloader(test_path, imgsz_test, batch_size*2, gs, opt,
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers)[0]  # testloader

        if not opt.resume:
            # 绘制数据集中的分类分布
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, save_dir=save_dir)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)
                if wandb:
                    wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.png')]})

            # Anchors
            # if not opt.noautoanchor:
            #     check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset 看md文件
    model.nc = nc  # attach number of classes to model 将定义的类别数设置到model中
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights 根据每个样本出现的频率计算每个类别的概率
    model.names = names 

    # Start training
    t0 = time.time()
    # 学习率上涨的训练次数
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class 创建一个空的map，用来存储每个样本的mAP
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move 计算其实的epoch
    scaler = amp.GradScaler(enabled=cuda)
    logger.info('Image sizes %g train, %g test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, save_dir, epochs))
    
    torch.save(model, wdir / 'init.pt') 
    
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # image_weights 参数用于按照图像难度和类别平衡进行带权重的图像采样，使训练过程更加关注难以检测的样本和稀有类别
            # Generate indices
            if rank in [-1, 0]:
                # 根据样本出现的频率和计算每个样本的mAP得到样本权重，即将少量样本、低map的样本提高采样的频率
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                # 更新dataset的索引，增加权重高的图片的出现频率
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                # 将计算好的权重同步到各个进程
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                # 使用 PyTorch 分布式通信将 rank 0 的 indices 广播给所有进程
                # 参数 0 表示从 rank 0 进程广播
                '''
                # 在调用 broadcast 之前：
                # rank 0: indices = torch.tensor([实际的索引数据])
                # rank 1: indices = torch.zeros(dataset.n)  # 全零张量
                # rank 2: indices = torch.zeros(dataset.n)  # 全零张量

                dist.broadcast(indices, 0)  # 广播操作

                # 在调用 broadcast 之后：
                # rank 0: indices = torch.tensor([实际的索引数据])  # 保持不变
                # rank 1: indices = torch.tensor([实际的索引数据])  # 已更新！
                # rank 2: indices = torch.tensor([实际的索引数据])  # 已更新！
                '''
                dist.broadcast(indices, 0)
                if rank != 0:
                    # 非主进程接收广播的数据并更新自己的 dataset.indices
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch) # 设置当前的训练轮次，确保随机数种子在每一轮训练都不一样
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start) 计算在第几个batch 全局
            '''
            non_blocking=False（默认）：同步传输，CPU 等待数据完全传输到 GPU 后才继续
            non_blocking=True：异步传输，CPU 立即继续执行，数据在后台传输
            '''
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                # 线性预热学习率
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                # 插值过程：
                # 当 ni = 0：accumulate = 1（不累积，每个批次都更新）
                # 当 ni = nw：accumulate = nbs / total_batch_size（正常累积步数）
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round()) # 在预热期间逐渐调整梯度累积步数
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    '''
                    j=0：pg0（其他参数，如 BN 层参数）
                    j=1：pg1（卷积权重）
                    j=2：pg2（偏置参数）

                    # 起始学习率
                    start_lr = hyp['warmup_bias_lr'] if j == 2 else 0.0
                    # 目标学习率  
                    end_lr = x['initial_lr'] * lf(epoch)
                    # 线性插值
                    x['lr'] = np.interp(ni, xi, [start_lr, end_lr])

                    偏置参数（j=2）：从 warmup_bias_lr（通常是 0.1）开始，逐渐降到目标学习率
                    其他参数（j=0,1）：从 0.0 开始，逐渐升到目标学习率
                    '''
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        '''
                        起始动量：warmup_momentum（通常较小，如 0.8）
                        目标动量：momentum（通常较大，如 0.937）
                        效果：预热期间使用较小动量，避免训练初期的不稳定
                        '''
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                # 是否开启多尺寸训练
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor 计算缩放因子
                if sf != 1:
                    # 线性缩放图片，同时保持图片的尺寸是gs的倍数
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            # 在 autocast 上下文中，PyTorch 会自动选择合适的数据类型（float16 或 float32）来执行运算，以提高训练速度和减少显存使用。
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward  # 部分使用 float16，部分使用 float32
                loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size # 自动选择精度
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                elif plots and ni == 3 and wandb:
                    wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('train*.jpg')]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema:
                ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                if epoch >= 3:
                    results, maps, times = test.test(opt.data,
                                                 batch_size=batch_size*2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 plots=plots and final_epoch,
                                                 log_imgs=opt.log_imgs if wandb else 0)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb:
                    wandb.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_p = fitness_p(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_r = fitness_r(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_ap50 = fitness_ap50(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_ap = fitness_ap(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if (fi_p > 0.0) or (fi_r > 0.0):
                fi_f = fitness_f(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            else:
                fi_f = 0.0
            if fi > best_fitness:
                best_fitness = fi
            if fi_p > best_fitness_p:
                best_fitness_p = fi_p
            if fi_r > best_fitness_r:
                best_fitness_r = fi_r
            if fi_ap50 > best_fitness_ap50:
                best_fitness_ap50 = fi_ap50
            if fi_ap > best_fitness_ap:
                best_fitness_ap = fi_ap
            if fi_f > best_fitness_f:
                best_fitness_f = fi_f

            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'best_fitness_p': best_fitness_p,
                            'best_fitness_r': best_fitness_r,
                            'best_fitness_ap50': best_fitness_ap50,
                            'best_fitness_ap': best_fitness_ap,
                            'best_fitness_f': best_fitness_f,
                            'training_results': f.read(),
                            'model': ema.ema.module.state_dict() if hasattr(ema, 'module') else ema.ema.state_dict(),
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            'wandb_id': wandb_run.id if wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if best_fitness == fi:
                    torch.save(ckpt, wdir / 'best_overall.pt')
                if best_fitness_p == fi_p:
                    torch.save(ckpt, wdir / 'best_p.pt')
                if best_fitness_r == fi_r:
                    torch.save(ckpt, wdir / 'best_r.pt')
                if best_fitness_ap50 == fi_ap50:
                    torch.save(ckpt, wdir / 'best_ap50.pt')
                if best_fitness_ap == fi_ap:
                    torch.save(ckpt, wdir / 'best_ap.pt')
                if best_fitness_f == fi_f:
                    torch.save(ckpt, wdir / 'best_f.pt')
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if ((epoch+1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if epoch >= (epochs-5):
                    torch.save(ckpt, wdir / 'last_{:03d}.pt'.format(epoch))
                elif epoch >= 420: 
                    torch.save(ckpt, wdir / 'last_{:03d}.pt'.format(epoch))
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = opt.name if opt.name.isnumeric() else ''
        fresults, flast, fbest = save_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'
        for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):
            if f1.exists():
                os.rename(f1, f2)  # rename
                if str(f2).endswith('.pt'):  # is *.pt
                    strip_optimizer(f2)  # strip optimizer
                    os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None  # upload
        # Finish
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb:
                wandb.log({"Results": [wandb.Image(str(save_dir / x), caption=x) for x in
                                       ['results.png', 'precision-recall_curve.png']]})
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    else:
        dist.destroy_process_group()

    wandb.run.finish() if wandb and wandb.run else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov4.weights', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    # WORLD_SIZE 参数主要用于分布式训练场景中，表示训练过程中总的进程数（通常对应使用的 GPU 数量）。当使用分布式数据并行 (DDP) 训练时，该参数帮助代码确定如何分配数据、同步梯度以及调整优化参数。如果没有设置，则默认为 1，即单 GPU 或单进程训练。
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    # 设置日志等级
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        cfg = opt.cfg if opt.cfg is not None else ''
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = cfg, ckpt, True
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    '''
    这段代码主要用于选择和配置设备，支持单卡和分布式（DDP）训练，具体流程如下：

    - **选择设备**  
    `device = select_device(opt.device, batch_size=opt.batch_size)` 调用 `select_device` 函数，根据传入的设备参数（例如 'cpu' 或 'cuda'）以及批量大小，自动选择合适的计算设备。

    - **分布式训练判断**  
    `if opt.local_rank != -1:` 判断是否处于分布式训练模式，如果 `local_rank` 不等于 -1，表示是使用分布式数据并行（DDP）训练。

    - **设置当前 GPU**  
    `assert torch.cuda.device_count() > opt.local_rank` 确保系统中有足够的 GPU，然后用  
    `torch.cuda.set_device(opt.local_rank)` 指定当前进程使用的 GPU，并将 `device` 更新为该 GPU。

    - **初始化分布式后端**  
    `dist.init_process_group(backend='nccl', init_method='env://')` 使用 NCCL 后端并通过环境变量进行初始化，建立各进程之间的通信。

    - **调整批量大小**  
    `assert opt.batch_size % opt.world_size == 0` 确保每个 GPU 得到的批量大小是整数，然后通过  
    `opt.batch_size = opt.total_batch_size // opt.world_size` 计算出每个 GPU 实际使用的批量大小。

    这段代码保证了在多 GPU 分布式训练模式下，每个进程能正确选定自己的 GPU，并通过 NCCL 通信来同步训练。
    '''
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    # 加载训练的超参数，一般就是学习率哪些的
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')

    # Train
    logger.info(opt)
    if not opt.evolve:
        # 如果不是在探索那种超参数更好则直接训练
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer, wandb)

    # Evolve hyperparameters (optional)
    else:
        # 这边就是在探索超参数了
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, wandb=wandb)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
