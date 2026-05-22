import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_backbone import resnet50_1d


class ASPPConv1D(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv1D, self).__init__(*modules)


class ASPPPooling1D(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling1D, self).__init__(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-1]
        for mod in self:
            x = mod(x)
        # Linear Interpolate 恢复长度
        return F.interpolate(x, size=size, mode='linear', align_corners=False)


class ASPP1D(nn.Module):
    """
    对应结构图中的 ASPP 1D 模块并联结构
    """

    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP1D, self).__init__()
        modules = []
        # ASPP_B1: 1x1 Conv
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # ASPP_B2, B3, B4: 3x3 Conv with Dilations
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv1D(in_channels, out_channels, rate))

        # ASPP_B5: Image Pooling
        modules.append(ASPPPooling1D(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 融合层: Concat 后的 Conv1d
        self.project = nn.Sequential(
            nn.Conv1d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # ASPP_Drop
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)  # Concat
        return self.project(res)


class DeepLabHead1D(nn.Sequential):
    """
    对应结构图中的 DeepLab Head 1D
    """

    def __init__(self, in_channels, num_classes):
        super(DeepLabHead1D, self).__init__(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1, bias=False),  # Head_Conv1
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_classes, kernel_size=1)  # Head_Conv2
        )


class FCNHead1D(nn.Sequential):
    """
    对应结构图中的 FCN Head (Aux) 1D
    """

    def __init__(self, in_channels, channels, num_classes):
        super(FCNHead1D, self).__init__(
            nn.Conv1d(in_channels, channels, kernel_size=3, padding=1, bias=False),  # FCN_Conv1
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # FCN_Drop
            nn.Conv1d(channels, num_classes, kernel_size=1)  # FCN_Conv2
        )


class DeepLabV3_1D(nn.Module):
    """
    将骨干网络和 Head 组合的最终 1D DeepLabV3
    """

    def __init__(self, in_channels=6, num_classes=2, aux_loss=True):
        super(DeepLabV3_1D, self).__init__()

        # 1. 主干网络
        self.backbone = resnet50_1d(in_channels=in_channels)

        # 2. 辅助分类头 (接在 Layer3 后面, ResNet50 的 Layer3 输出通道为 1024)
        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_classifier = FCNHead1D(in_channels=1024, channels=256, num_classes=num_classes)

        # 3. ASPP (接在 Layer4 后面, ResNet50 的 Layer4 输出通道为 2048)
        # 根据你的 Graphviz 图，原生设计使用的是 12, 24, 36
        # 注意: 如果序列长度较短，建议按照你文档的建议修改此处为 [2, 5, 9] 或其他互质数
        atrous_rates = [12, 24, 36]
        self.aspp = ASPP1D(in_channels=2048, atrous_rates=atrous_rates)

        # 4. 主分类头
        self.classifier = DeepLabHead1D(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        input_shape = x.shape[-1]

        # 获取骨干网络提取的特征
        features_layer3, features_layer4 = self.backbone(x)

        result = {}

        # 主分支预测 (ASPP -> DeepLabHead -> Interpolate)
        x_main = self.aspp(features_layer4)
        x_main = self.classifier(x_main)
        x_main = F.interpolate(x_main, size=input_shape, mode='linear', align_corners=False)
        result["out"] = x_main

        # 辅助分支预测 (Layer3 -> FCNHead -> Interpolate)
        if self.aux_loss:
            x_aux = self.aux_classifier(features_layer3)
            x_aux = F.interpolate(x_aux, size=input_shape, mode='linear', align_corners=False)
            result["aux"] = x_aux

        return result