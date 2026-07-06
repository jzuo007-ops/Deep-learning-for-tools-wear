from .bilstm_seg import BiLSTMSeg
from .deeplabv3_model import DeepLabV3_1D
from .tcn_seg import TCNSeg
from .unet_1d import UNet1D


SEGMENTATION_MODEL_NAMES = (
    "deeplabv3_1d",
    "unet_1d",
    "tcn_seg",
    "bilstm_seg",
)


def build_segmentation_model(
    name,
    in_channels=7,
    num_classes=3,
    aux_loss=False,
    backbone_name="resnet50",
    **kwargs,
):
    name = name.lower()
    if name in {"deeplabv3", "deeplabv3_1d", "deeplab"}:
        return DeepLabV3_1D(
            in_channels=in_channels,
            num_classes=num_classes,
            aux_loss=aux_loss,
            classification=False,
            backbone_name=backbone_name,
        )
    if name in {"unet", "unet_1d", "u-net"}:
        return UNet1D(
            in_channels=in_channels,
            num_classes=num_classes,
            aux_loss=aux_loss,
            **kwargs,
        )
    if name in {"tcn", "tcn_seg", "tcn_1d"}:
        return TCNSeg(
            in_channels=in_channels,
            num_classes=num_classes,
            aux_loss=aux_loss,
            **kwargs,
        )
    if name in {"bilstm", "bilstm_seg", "bilstm_1d"}:
        return BiLSTMSeg(
            in_channels=in_channels,
            num_classes=num_classes,
            aux_loss=aux_loss,
            **kwargs,
        )
    raise ValueError(f"Unknown segmentation model {name!r}; expected one of {SEGMENTATION_MODEL_NAMES}.")
