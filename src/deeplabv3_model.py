import torch
import torch.nn as nn
import torch.nn.functional as F
from .lstm_backbone import lstm_backbone_1d
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
        return F.interpolate(x, size=size, mode='linear', align_corners=False)


class ASPP1D(nn.Module):
    """
    对应结构图中的 ASPP 1D 模块并联结构
    """

    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP1D, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv1D(in_channels, out_channels, rate))

        modules.append(ASPPPooling1D(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv1d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHead1D(nn.Sequential):
    """
    对应结构图中的 DeepLab Head 1D
    """

    def __init__(self, in_channels, num_classes):
        super(DeepLabHead1D, self).__init__(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_classes, kernel_size=1)
        )


class FCNHead1D(nn.Sequential):
    """
    对应结构图中的 FCN Head (Aux) 1D
    """

    def __init__(self, in_channels, channels, num_classes):
        super(FCNHead1D, self).__init__(
            nn.Conv1d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(channels, num_classes, kernel_size=1)
        )


class DeepLabV3_1D(nn.Module):
    """
    将骨干网络和 Head 组合的最终 1D DeepLabV3。
    为分类任务保留原始结构，并通过时序池化得到样本级别 logits。
    """

    def __init__(
        self,
        in_channels=6,
        num_classes=2,
        aux_loss=True,
        classification=False,
        backbone_name="resnet50",
    ):
        super(DeepLabV3_1D, self).__init__()

        self.backbone_name = backbone_name
        if backbone_name == "resnet50":
            self.backbone = resnet50_1d(in_channels=in_channels)
        elif backbone_name == "lstm":
            self.backbone = lstm_backbone_1d(in_channels=in_channels)
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.aux_loss = aux_loss
        self.classification = classification
        if aux_loss:
            self.aux_classifier = FCNHead1D(in_channels=1024, channels=256, num_classes=num_classes)

        atrous_rates = [12, 24, 36]
        self.aspp = ASPP1D(in_channels=2048, atrous_rates=atrous_rates)
        self.classifier = DeepLabHead1D(in_channels=256, num_classes=num_classes)

    def _pool_logits(self, x):
        if self.classification:
            return F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return x

    def forward(self, x):
        input_shape = x.shape[-1]
        features_layer3, features_layer4 = self.backbone(x)

        result = {}

        x_main = self.aspp(features_layer4)
        x_main = self.classifier(x_main)
        if self.classification:
            x_main = self._pool_logits(x_main)
        else:
            x_main = F.interpolate(x_main, size=input_shape, mode='linear', align_corners=False)
        result["out"] = x_main

        if self.aux_loss:
            x_aux = self.aux_classifier(features_layer3)
            if self.classification:
                x_aux = self._pool_logits(x_aux)
            else:
                x_aux = F.interpolate(x_aux, size=input_shape, mode='linear', align_corners=False)
            result["aux"] = x_aux

        return result
