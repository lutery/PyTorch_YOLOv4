import torch.nn.functional as F

from utils.general import *

import torch
from torch import nn

try:
    from mish_cuda import MishCuda as Mish
    
except:
    class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
        def forward(self, x):
            return x * F.softplus(x).tanh()


class Reorg(nn.Module):
    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


def make_divisible(v, divisor):
    # Function ensures all layers have a channel number that is divisible by 8
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    return math.ceil(v / divisor) * divisor


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class FeatureConcat(nn.Module):
    def __init__(self, layers):
        '''
        输入的layers是一个列表，表示需要拼接的特征图的索引
        例如，layers = [0, 1, 2]表示需要拼接第0、1、2个特征图
        '''
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag # 是否有多个层需要拼接

    def forward(self, x, outputs):
        '''
        param x: 输入的特征图
        param outputs: 输出的特征图 这个应该时目前已经计算好的特征图
        '''
        # 从已经计算好的特征图中取出需要拼接的特征图
        # 如果有多个层需要拼接，则将它们拼接在一起
        # 否则直接返回第一个特征图
        # 这里的拼接是沿着通道的维度进行拼接
        # 例如，输入的特征图是[1, 3, 224, 224]，表示batch_size=1，通道数=3，宽高=224
        # 如果layers=[0, 1, 2]，则拼接后的特征图是[1, 9, 224, 224]
        # 如果layers=[0]，则返回[1, 3, 224, 224]
        '''
        为什么这里不使用detach()？todo
        不太清楚，可能实际使用时却是不需要使用，不过由于cfg中没有使用FeatureConcat2和FeatureConcat3，也不太好对比
        '''
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class FeatureConcat2(nn.Module):
    '''
    与FeatureConcat类似，但是只拼接两个特征图
    这个类主要是为了在拼接两个特征图时，避免使用FeatureConcat类的多余操作
    '''
    def __init__(self, layers):
        super(FeatureConcat2, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        '''
        为什么使用detach()？
        因为在拼接特征图时，可能会导致梯度计算错误
        例如，假设有两个特征图A和B，A的梯度是dA，B的梯度是dB
        如果直接拼接它们，得到的特征图C的梯度是dC
        dC = dA + dB
        但是如果在拼接时，B的梯度被计算了，那么dC就会变成dA + dB + dB
        这样就会导致梯度计算错误
        所以在拼接时，将B的梯度计算关闭，使用detach()函数
        这样就不会计算B的梯度了
        这样就可以避免梯度计算错误
        
        '''
        return torch.cat([outputs[self.layers[0]], outputs[self.layers[1]].detach()], 1)


class FeatureConcat3(nn.Module):
    '''
    与FeatureConcat类似，但是只拼接三个特征图
    这个类主要是为了在拼接两个特征图时，避免使用FeatureConcat类的多余操作
    '''
    def __init__(self, layers):
        super(FeatureConcat3, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[self.layers[0]], outputs[self.layers[1]].detach(), outputs[self.layers[2]].detach()], 1)


class FeatureConcat_l(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat_l, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[i][:,:outputs[i].shape[1]//2,:,:] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]][:,:outputs[self.layers[0]].shape[1]//2,:,:]


class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class MixConv2d(nn.Module):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        '''
        多尺度卷积，这段代码属于多尺度卷积层（MixConv2d）的构造部分，主要作用是将总输出通道数（out_ch）按两种策略分成多个组，每个组对应一个不同的卷积核大小，从而实现多尺度特征提取。具体过程如下
        是一种 混合深度卷积（Mixed Depthwise Convolution），其核心思想是通过使用不同大小的卷积核（kernel size）来提取多尺度特征
        这种卷积主要用于需要捕获多尺度特征的场景，比如目标检测、语义分割等任务

        params in_ch: int, 输入通道数
        params out_ch: int, 输出通道数
        params k: tuple, 卷积核大小
        params stride: int, 步长
        params dilation: int, 膨胀率
        params bias: bool, 是否使用偏置
        params method: str, 'equal_params' or 'equal_ch', 每个组的参数数量相等或每个组的通道数相等
        '''
        super(MixConv2d, self).__init__()

        groups = len(k)
        # todo 详细了解这边的操作
        if method == 'equal_ch':  # equal channels per group
            # 将输出通道数分成groups组，每组的输出通道数相等
            # groups - 1E-6防止除0
            # torch.linspace(0, groups - 1E-6, out_ch)生成一个从0到groups-1E-6的等差数列，即有groups个数值
            # floor()函数向下取整，得到每个组的索引
            i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            # 计算每个输出通道的通道数，因为i输出的格式如下：
            '''
            tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
            2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
            3., 3.])

            所以i==g实在判断上面的矩阵中每个元素是否等于当前组索引，等于为true，不等于为false
            然后true=1，false=0，使用sum()函数将每组的true相加，得到每组的通道数
            '''
            ch = [(i == g).sum() for g in range(groups)]
            # [tensor(32), tensor(32), tensor(32), tensor(32)]
        else:  # 'equal_params': equal parameter count per group
            # 按每个卷积核的参数量相等进行分组
            # 这里就是在按照参数量（通道数 × 卷积核大小²）相等进行
            # 计算，手算很简单，但是代码实现起来就比较复杂了 todo 了解其中的代数
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        # 构建了多个卷积，后续使用时将它们的输出拼接在一起
        # 每个卷积的输入通道、步长、dilation、偏置相同
        # 每个卷积的输出通道数、卷积核大小不同、填充率也不同
        # 这样就保证了每个卷积核输出的特征图大小相同，可以拼接在一起
        self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch,
                                          out_channels=ch[g],
                                          kernel_size=k[g],
                                          stride=stride,
                                          padding=k[g] // 2,  # 'same' pad
                                          dilation=dilation,
                                          bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)


# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)  # sigmoid(ctx)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class HardSwish(nn.Module):  # https://arxiv.org/pdf/1905.02244.pdf
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
            是一种实现 可变形卷积（Deformable Convolution） 的模块。它通过学习卷积核的偏移量，使卷积操作能够动态调整采样位置，从而增强模型对几何变形的建模能力
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # 一个普通的卷积
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # 偏移量卷积 todo
        # 输出通道数为 2 * kernel_size * kernel_size，表示每个卷积核位置的偏移量（x 和 y 坐标）。
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        # 偏移量初始化为 0
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            # 调制机制（可选）
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
    
    
class GAP(nn.Module):
    def __init__(self):
        super(GAP, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        #b, c, _, _ = x.size()        
        return self.avg_pool(x)#.view(b, c)
    
    
class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()
    def forward(self, x):    
        return x


class ScaleChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(ScaleChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x.expand_as(a) * a


class ScaleSpatial(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(ScaleSpatial, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return x * a
