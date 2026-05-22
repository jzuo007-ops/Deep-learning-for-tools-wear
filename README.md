#项目目录

├── src/
│ ├── __init__.py
│ ├── deeplabv3_model.py # DeepLabV3 模型定义     y
│ ├── mobilenet_backbone.py # MobileNet 骨干网络  n
│ └── resnet_backbone.py # ResNet 骨干网络        y
├── train_utils/   n
│ ├── __init__.py
│ ├── distributed_utils.py # 分布式训练工具
│ └── train_and_eval.py # 训练与评估函数
├── deeplabv3_resnet50.png # 模型架构图
├── get_palette.py # 调色板生成
├── my_dataset.py # 自定义数据集
├── palette.json # 调色板配置
├── pascal_voc_classes.json # Pascal VOC 类别映射
├── predict.py # 预测/推理脚本
├── README.md # 项目说明
├── requirements.txt # 依赖包清单
├── results20211027-104607.txt # 训练结果记录
├── train.py # 单 GPU 训练脚本
├── train_multi_GPU.py # 多 GPU 训练脚本
├── transforms.py # 数据预处理/增强
└── validation.py # 验证脚本
