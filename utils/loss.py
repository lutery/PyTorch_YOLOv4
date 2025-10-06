# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    # Focal Loss 是一种改进的交叉熵损失函数，专门用于解决类别不平衡问题，特别是在目标检测任务中。
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss() 基础损失函数
        self.gamma = gamma # 聚焦参数，控制难易样本的权重差异
        self.alpha = alpha # 平衡参数，控制正负样本的权重比例
        self.reduction = loss_fcn.reduction # 保留基础损失函数的reduction方式
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element 需要对每个元素单独计算，所以这里强制去掉reduction

    def forward(self, pred, true):
        # 计算基础的BCE损失
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        # 计算预测概率
        pred_prob = torch.sigmoid(pred)  # prob from logits # 将 logits 转换为概率 [0, 1]
        
        # 3. 计算 p_t（正确分类的概率）
        # 如果 true=1（正样本）：p_t = pred_prob
        # 如果 true=0（负样本）：p_t = 1 - pred_prob
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)

        # 4. 计算 alpha 因子
        # 正样本使用 alpha，负样本使用 1-alpha
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)

        # 5. 计算调制因子（Focal Loss 的核心）
        # (1 - p_t)^gamma：p_t 越大（越容易分类），权重越小
        modulating_factor = (1.0 - p_t) ** self.gamma

        # # 6. 最终损失
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(p, targets, model):  # predictions, targets, model
    '''
    p:  这里pred是一个list，包含三个yolo层的输出， 每一层的shape=(bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
    targets: targets.shape = [N, 6]，其中6=(image, class, x, y, w, h)
    model: model
    '''
    device = targets.device
    #print(device)
    # todo 这几个的作用
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device) 
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets 筛选与目标框匹配的anchors，扩展正样本
    h = model.hyp  # hyperparameters

    # Define criteria 定于分类损失和目标损失
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    # 标签平滑，防止过拟合，避免硬编码导致过于自信
    # 但是这里没有使用，传入的是0.0
    # cp=1.0, cn=0.0是传统的one-hot编码
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    balance = [4.0, 1.0, 0.5, 0.4, 0.1] if no == 5 else balance
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no >= 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
    '''
    p:  这里pred是一个list，包含三个yolo层的输出， 每一层的shape=(bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
    targets: targets.shape = [N, 6]，其中6=(image, class, x, y, w, h)，其中的xywh是相对于整张图像的归一化坐标，image时图片的索引
    model: model
    '''
    device = targets.device
    nt = targets.shape[0]  # number of anchors, targets  获取当前batch中的所有目标
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain  todo 作用
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets todo作用 # 四个方向的偏移

    g = 0.5  # offset 偏移阈值
    multi_gpu = is_parallel(model) # 是否多GPU训练，因为多GPU时，model会被封装成nn.DataParallel或nn.DistributedDataParallel，所以要获取yolo_layers需要从不同成员变量获取出来
    # yolo_layers表示yolo层的索引，遍历每一层的yolo
    for i, jj in enumerate(model.module.yolo_layers if multi_gpu else model.yolo_layers):
        # get number of grid points and anchor vec for this yolo layer 获取指定层的yolo的anchors
        anchors = (model.module.module_list[jj].anchor_vec if multi_gpu else model.module_list[jj].anchor_vec).to(device)
        # p[i]：获取指定层的yolo输出
        # p[i].shape = (bs, anchors, grid_h, grid_w, classes + xywh)
        # [3, 2, 3, 2]：获取指定层的yolo输出的grid_h和grid_w [grid_w, grid_h, grid_w, grid_h]
        # gain = [1, 1, grid_w, grid_h, grid_w, grid_h]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        # targets * gain：是将归一化坐标转换为网格坐标系的关键操作， shape is  [N, image, class, x, y, w, h]，也就是t
        a, t, offsets = [], targets * gain, 0
        if nt: # 如果有目标
            na = anchors.shape[0]  # number of anchors 当前yolo层的anchor数量，默认是3
            # torch.arange(na)：生成一个从0到na-1的张量，shape =(na,)
            # .view(na, 1)：将其变形为(na, 1)
            # .repeat(1, nt)：将其在第二个维度上重复nt次，变为(na, nt)，其中nt代表target中有多少个目标，这样子为每一个anchors都分配了nt个目标，也就是所有的target都会去和每一个anchor去计算iou
            # .to(device)：将其移动到指定设备上
            # at shape (na(多少个anchors), nt（目标数量）)
            at = torch.arange(na).view(na, 1).repeat(1, nt).to(device)  # anchor tensor, same as .repeat_interleave(nt)
            # t[:, 4:6]: 目标的宽高（网格坐标系），shape = (n, 2)
            # anchors: 当前层的anchor尺寸，shape = (3, 2)
            # r: 目标宽高与anchor宽高的比值，shape = (3, n, 2)
            # 这里就对应着提取与anchors比值最大的anchor作为目标框的匹配anchor
            # t[None, :, 4:6]      # shape = (1, n, 2)，添加维度用于广播
            # anchors[:, None]     # shape = (3, 1, 2)，添加维度用于广播 None在哪里说明在哪里进行了维度扩展为1
            r = t[None, :, 4:6] / anchors[:, None]  # wh ratio (3, n, 2)  wh是相对于整张图像的归一化坐标，除以anchors后，变成了相对于anchors的归一化坐标
            # torch.max(r, 1. / r).max(2)返回的是宽宽/高高比中的最大值以及其位置索引。max(2)代表在第2个维度上取最大值，这样整体张量的维度就从（3，n, 2)塌陷到(3, n)
            # [0]表示取最大值，shape = (3, n)
            # torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t'] 后的shape is （3（na）, n）boolean矩阵
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare 根据上一步中计算高与高的比值，宽与宽的比值，对比阈值，筛选出匹配的 anchor 索引
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2)) # 这里是yolov3时候计算面积占比IOU来筛选匹配的anchor
            # at[j]：代表从at中筛选出与目标匹配的anchor索引，shape is (M,) 匹配上的anchor数量，使用j进行筛选最后会将j中为True的部分筛选出来，那么j所在的维度就会被压缩掉
            # t.repeat(na, 1, 1)：将t在第一个维度上重复na次，shape is (na, n, 6（image, class, x, y, w, h）)
            # t.repeat(na, 1, 1)[j]：代表从t中筛选出与目标匹配的目标信息，shape is (M, 6（image, class, x, y, w, h）)
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter  a 是匹配上的 anchor 索引/t 是筛选出来与 anchor 匹配的真实目标信息

            # overlaps
            gxy = t[:, 2:4]  # grid xy 获取匹配目标框的中心点坐标，shape is (M, 2)
            z = torch.zeros_like(gxy) # z shape is (M, 2)，全0张量 零偏移
            # 如果目标中心点靠近网格边界，将该目标也分配给相邻的网格，增加正样本数量。
            # gxy % 1.  # 获取小数部分，表示在网格内的相对位置
            # 例如：gxy = [13.3, 7.2]
            # gxy % 1. = [0.3, 0.2]
            # gxy % 1. < g  # 小数部分 < 0.5，说明靠近网格左侧或上侧 # [0.3 < 0.5, 0.2 < 0.5] = [True, True]
            # # 坐标 > 1，确保不是第0个网格（避免越界） # [13.3 > 1, 7.2 > 1] = [True, True]
            # # 两个条件同时满足# shape = (M, 2)，表示 [x方向是否满足, y方向是否满足] 得到一个布尔矩阵 【True, True】表示x和y方向都满足
            # # 转置后分别赋值给 j 和 k
            # # j: x方向满足条件的目标，shape = (M,)
            # k: y方向满足条件的目标，shape = (M,)
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            # 小数部分 > 0.5，说明靠近网格右侧或下侧
            # [0.3 > 0.5, 0.2 > 0.5] = [False, False]
            # gain[[2, 3]] - 1.  # 网格最大索引 = [grid_w - 1, grid_h - 1] # 例如：26x26 网格 -> [25, 25]
            #  # 确保不是最后一个网格（避免越界） # [13.3 < 25, 7.2 < 25] = [True, True]
            #  # 转置后分别赋值
            # # l: x方向满足条件的目标，shape = (M,)
            # m: y方向满足条件的目标，shape = (M,)
            l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
            # 将筛选出来的新增正样本组合起来，扩展 anchor 和目标信息
            '''
            # 原始匹配结果
            a.shape = (M,)      # M 个匹配的 anchor 索引
            t.shape = (M, 6)    # M 个匹配的目标信息

            # 扩展后
            a_new = torch.cat((
                a,      # 原始目标
                a[j],   # 靠近左侧的目标（额外分配给左侧网格）
                a[k],   # 靠近上侧的目标（额外分配给上侧网格）
                a[l],   # 靠近右侧的目标（额外分配给右侧网格）
                a[m]    # 靠近下侧的目标（额外分配给下侧网格）
            ), 0)

            # 新的数量 ≤ M * 5（最多5倍，实际上会少一些）
            '''
            a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
            # 第四行：计算偏移量
            '''
            off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]])

            offsets = torch.cat((
                z,                # [0, 0]，原始网格，无偏移
                z[j] + off[0],    # [1, 0]，向右偏移
                z[k] + off[1],    # [0, 1]，向下偏移
                z[l] + off[2],    # [-1, 0]，向左偏移
                z[m] + off[3]     # [0, -1]，向上偏移
            ), 0) * g  # 乘以 0.5，得到实际偏移量

            # offsets.shape = (扩展后的数量, 2)
            '''
            offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        b, c = t[:, :2].long().T  # image, class 获取图片索引和类别索引，shape is (M,)  M代表匹配上的目标数量
        gxy = t[:, 2:4]  # grid xy 获取匹配目标框的中心点坐标，shape is (M, 2)
        gwh = t[:, 4:6]  # grid wh 获取匹配目标框的宽高，shape is (M, 2)
        gij = (gxy - offsets).long() # 正式对坐标进行偏移，增加样本数量，其中使用long取整
        gi, gj = gij.T  # grid xy indices 得到网格索引，shape is (M,)  M代表匹配上的目标数量

        # Append
        indices.append((b, a, gj, gi))  # image, anchor, grid indices # b: 图片索引, a: anchor索引, gj: 网格y索引, gi: 网格x索引组合
        # indices.append((b, a, gj.clamp_(0, int(gain[3] - 1)), gi.clamp_(0, int(gain[2] - 1))))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 是在构建目标框的相对坐标表示，xy是相对于网格左上角的偏移，wh是宽高 shape = (M, 4)，格式为 [相对x, 相对y, w, h]
        anch.append(anchors[a])  # anchors 提取匹配的anchor，shape = (M, 2)
        tcls.append(c)  # class 提取目标类别，shape = (M,)

    # 返回所有yolo层的目标信息
    # tcls： list of (M,)  每个元素是一个张量，包含该层所有匹配目标的类别索引
    # tbox： list of (M, 4) 每个元素是一个张量，包含该层所有匹配目标的边界框，格式为 [相对x, 相对y, w, h]
    # indices： list of (b, a, gj, gi) 每个元素是一个元组，包含该层所有匹配目标的图片索引、anchor索引、网格y索引和网格x索引
    # anch： list of (M, 2) 每个元素是一个张量，包含该层所有匹配目标的anchor尺寸，格式为 [anchor_w, anchor_h]
    # todo，后续要查看损失如何计算相同anchor去计算匹配不同中心点的目标
    return tcls, tbox, indices, anch
