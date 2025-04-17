"""
Implementation of 1D version of StarNet for vector data classification.

This is a modified version of StarNet adapted for 1D vector inputs with length 16.
The network performs classification tasks on 1D vector data.

Created based on: Xu Ma's StarNet 
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

model_urls = {
    # 一维模型暂无预训练权重
}


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm1d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 3, 1, (3 - 1) // 2, groups=dim, with_bn=True)  # 使用较小的卷积核
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 3, 1, (3 - 1) // 2, groups=dim, with_bn=False)  # 使用较小的卷积核
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet1D(nn.Module):
    def __init__(self, base_dim=16, depths=[2, 2, 4, 2], mlp_ratio=2, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 16  # 初始通道数调整为16
        
        # 输入向量长度为16，通道数为1
        # stem layer - 对于短序列，使用较小的stride以保留更多信息
        self.stem = nn.Sequential(ConvBN(1, self.in_channel, kernel_size=3, stride=1, padding=1), nn.ReLU6())
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth
        
        # build stages - 由于输入序列较短，减少下采样次数和强度
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** min(i_layer, 2)  # 限制通道数增长
            # 对于短序列，第一个阶段不下采样，后续阶段使用较小的stride
            stride = 1 if i_layer == 0 else 2
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, stride, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
            
        # head
        self.norm = nn.BatchNorm1d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 输入x的形状应为 [batch_size, 1, 16]
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


@register_model
def starnet1d_s1(pretrained=False, **kwargs):
    model = StarNet1D(16, [1, 1, 2, 1], mlp_ratio=2, num_classes=kwargs.get('num_classes', 1000))
    return model


@register_model
def starnet1d_s2(pretrained=False, **kwargs):
    model = StarNet1D(24, [1, 2, 3, 1], mlp_ratio=2, num_classes=kwargs.get('num_classes', 1000))
    return model


@register_model
def starnet1d_s3(pretrained=False, **kwargs):
    """较大模型，适合16长度的一维向量"""
    model = StarNet1D(32, [2, 2, 4, 2], mlp_ratio=3, num_classes=kwargs.get('num_classes', 1000))
    return model


# # 使用示例
# def example_usage():
#     # 创建模型
#     model = starnet1d_s1(num_classes=10)  # 假设有10个分类
    
#     # 创建一个随机输入张量 [batch_size, channels, length]
#     batch_size = 4
#     x = torch.randn(batch_size, 1, 16)
    
#     # 前向传播
#     output = model(x)
    
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
    
#     return model


# if __name__ == "__main__":
#     model = example_usage()


