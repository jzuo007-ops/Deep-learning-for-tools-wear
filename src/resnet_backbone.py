import torch
import torch.nn as nn


class Bottleneck1D(nn.Module):
    """
    对应结构图中的 Bottleneck 内部结构详情
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck1D, self).__init__()
        # B1/B2_C1: Conv1d(k=1)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        # B1/B2_C2: Conv1d(k=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        # B1/B2_C3: Conv1d(k=1)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 对应结构图中的 Shortcut (投影或恒等)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # Bottleneck 1: 投影 Shortcut

        out += identity  # B1/B2_Add
        out = self.relu(out)  # B1/B2_Out

        return out


class ResNet50_1D(nn.Module):
    """
    对应结构图中的 ResNet50 1D Backbone
    """

    def __init__(self, in_channels=6, replace_stride_with_dilation=None):
        super(ResNet50_1D, self).__init__()

        if replace_stride_with_dilation is None:
            # 默认：Layer 3 和 Layer 4 使用膨胀卷积代替下采样 (DeepLab 系列常规操作)
            replace_stride_with_dilation = [False, True, True]

        self.inplanes = 64
        self.dilation = 1

        # Conv1: Conv1d(k=7, s=2, p=3)
        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # MaxPool: MaxPool1d(k=3, s=2, p=1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Layer 1~4: 按照标准 ResNet50 的 block 数量 (3, 4, 6, 3)
        self.layer1 = self._make_layer(Bottleneck1D, 64, 3)
        self.layer2 = self._make_layer(Bottleneck1D, 128, 4, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(Bottleneck1D, 256, 6, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(Bottleneck1D, 512, 3, stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        low_level_features = x  # 供 FCN Aux 使用
        x = self.layer4(x)  # 供 ASPP 使用

        return low_level_features, x


def resnet50_1d(in_channels=6, **kwargs):
    return ResNet50_1D(in_channels=in_channels, **kwargs)