# 一维deeplabv3

Graphviz结构图

digraph DeepLabV3_1D_Clean {
    // 全局设定：直角连线，严格的节点距离控制
    graph [rankdir=TB, splines=ortho, nodesep=0.5, ranksep=0.6, fontname="Helvetica"];
    node [shape=box, style="rounded,filled", fillcolor=white, fontname="Helvetica", fontsize=11, margin=0.15];
    edge [fontname="Helvetica", fontsize=10, color="#444444", penwidth=1.2];

    // ==========================================
    // 核心输入与主干道 (强制权重为100，确保绝对垂直对齐)
    // ==========================================
    Input [label="Input\nShape: (6, L)", fillcolor="#e6f2ff", color="#0066cc", penwidth=2];
    
    subgraph cluster_backbone {
        label="ResNet50 1D Backbone";
        style=dashed; color="#888888"; bgcolor="#fafafa";
        
        Conv1 [label="Conv1d(k=7, s=2, p=3)\nBN, ReLU"];
        MaxPool [label="MaxPool1d(k=3, s=2, p=1)"];
        Layer1 [label="Layer 1\nBottleneck1 x 1\nBottleneck2 x 2"];
        Layer2 [label="Layer 2\nBottleneck1 x 1\nBottleneck2 x 3"];
        Layer3 [label="Layer 3\nBottleneck1 x 1\nBottleneck2 x 5\nDilations (r=1, r=2)"];
        Layer4 [label="Layer 4\nBottleneck1 x 1\nBottleneck2 x 2\nDilations (r=2, r=4)"];
    }
    
    // 强制南出北进，超高权重保证主轴笔直
    Input:s -> Conv1:n [weight=100];
    Conv1:s -> MaxPool:n [weight=100];
    MaxPool:s -> Layer1:n [weight=100];
    Layer1:s -> Layer2:n [weight=100];
    Layer2:s -> Layer3:n [weight=100];
    Layer3:s -> Layer4:n [weight=100];
    
    // ==========================================
    // FCN Head 辅助分支 (强制定向到右侧)
    // ==========================================
    subgraph cluster_fcn {
        label="FCN Head (Aux) 1D";
        style=solid; color="#cc9900"; bgcolor="#fff9e6";
        
        FCN_Conv1 [label="Conv1d(k=3, s=1, p=1)\nBN, ReLU"];
        FCN_Drop [label="Dropout"];
        FCN_Conv2 [label="Conv1d(k=1, s=1)"];
        FCN_Interp [label="Linear Interpolate"];
        Aux_Out [label="Aux Output\n(num_classes, L)", fillcolor="#e6ffe6", color="#00aa00", penwidth=2];
    
        FCN_Conv1:s -> FCN_Drop:n [weight=10];
        FCN_Drop:s -> FCN_Conv2:n [weight=10];
        FCN_Conv2:s -> FCN_Interp:n [weight=10];
        FCN_Interp:s -> Aux_Out:n [weight=10];
    }
    
    // 从Layer3的东侧(e)出发，进入FCN_Conv1的北侧(n)，避开主干道连线
    Layer3:e -> FCN_Conv1:n [xlabel=" Aux 分支 ", weight=1];
    
    // ==========================================
    // ASPP 模块并联结构
    // ==========================================
    subgraph cluster_aspp {
        label="ASPP 1D";
        style=solid; color="#0066cc"; bgcolor="#eef9ff";
        
        { rank=same; ASPP_B1; ASPP_B2; ASPP_B3; ASPP_B4; ASPP_B5; }
    
        ASPP_B1 [label="Conv1d\n(k=1, s=1)"];
        ASPP_B2 [label="Conv1d\n(k=3, s=1, r=12)"];
        ASPP_B3 [label="Conv1d\n(k=3, s=1, r=24)"];
        ASPP_B4 [label="Conv1d\n(k=3, s=1, r=36)"];
        ASPP_B5 [label="AdaptiveAvgPool1d(1)\nConv1d(k=1, s=1)\nLinear Interp"];
        
        Concat [label="Concat", fillcolor="#ffe6cc", shape=folder];
        ASPP_Out [label="Conv1d(k=1, s=1)\nBN, ReLU"];
        ASPP_Drop [label="Dropout"];
    
        // 合并到 Concat 容器
        ASPP_B1:s -> Concat:n;
        ASPP_B2:s -> Concat:n;
        ASPP_B3:s -> Concat:n;
        ASPP_B4:s -> Concat:n;
        ASPP_B5:s -> Concat:n;
        
        Concat:s -> ASPP_Out:n [weight=100];
        ASPP_Out:s -> ASPP_Drop:n [weight=100];
    }
    
    // 从Layer4统一分配到ASPP各分支
    Layer4:s -> ASPP_B1:n;
    Layer4:s -> ASPP_B2:n;
    Layer4:s -> ASPP_B3:n;
    Layer4:s -> ASPP_B4:n;
    Layer4:s -> ASPP_B5:n;
    
    // ==========================================
    // DeepLab Main Head 主分支
    // ==========================================
    subgraph cluster_head {
        label="DeepLab Head 1D";
        style=solid; color="#cc9900"; bgcolor="#fff9e6";
        
        Head_Conv1 [label="Conv1d(k=3, s=1, p=1)\nBN, ReLU"];
        Head_Conv2 [label="Conv1d(k=1, s=1)"];
        Head_Interp [label="Linear Interpolate"];
        Main_Out [label="Main Output\n(num_classes, L)", fillcolor="#e6ffe6", color="#00aa00", penwidth=2];
    
        Head_Conv1:s -> Head_Conv2:n [weight=10];
        Head_Conv2:s -> Head_Interp:n [weight=10];
        Head_Interp:s -> Main_Out:n [weight=10];
    }
    ASPP_Drop:s -> Head_Conv1:n [weight=100];
    
    // ==========================================
    // Bottleneck 细节结构 (独立图例区，完全断开物理连线干扰)
    // ==========================================
    subgraph cluster_bottlenecks {
        label="Bottleneck 内部结构详情";
        style=dashed; color="#aaaaaa";
        
        // 并排显示两个 Bottleneck
        subgraph cluster_bot1 {
            label="Bottleneck 1 (带投影 Shortcut)";
            style=solid; color="#55aa00"; bgcolor="#f4ffe8";
            
            B1_In [label="Input", shape=plaintext];
            B1_C1 [label="Conv1d(k=1, s=stride)\nBN, ReLU"];
            B1_C2 [label="Conv1d(k=3, s=1, r)\nBN, ReLU"];
            B1_C3 [label="Conv1d(k=1, s=1)\nBN"];
            B1_Proj [label="Conv1d(k=1, s=stride)\nBN"];
            B1_Add [label="+", shape=circle, fixedsize=true, width=0.4];
            B1_Out [label="ReLU"];
    
            // 内部严格罗盘定位
            B1_In:s -> B1_C1:n [weight=10];
            B1_C1:s -> B1_C2:n [weight=10];
            B1_C2:s -> B1_C3:n [weight=10];
            B1_C3:s -> B1_Add:n [weight=10];
            
            // 侧边投影分支
            B1_In:e -> B1_Proj:n [weight=1];
            B1_Proj:s -> B1_Add:e [weight=1];
            
            B1_Add:s -> B1_Out:n [weight=10];
        }
    
        subgraph cluster_bot2 {
            label="Bottleneck 2 (恒等 Shortcut)";
            style=solid; color="#55aa00"; bgcolor="#f4ffe8";
            
            B2_In [label="Input", shape=plaintext];
            B2_C1 [label="Conv1d(k=1, s=1)\nBN, ReLU"];
            B2_C2 [label="Conv1d(k=3, s=1, r)\nBN, ReLU"];
            B2_C3 [label="Conv1d(k=1, s=1)\nBN"];
            B2_Id [label="Identity\n(直接连接)", shape=plaintext];
            B2_Add [label="+", shape=circle, fixedsize=true, width=0.4];
            B2_Out [label="ReLU"];
    
            // 内部严格罗盘定位
            B2_In:s -> B2_C1:n [weight=10];
            B2_C1:s -> B2_C2:n [weight=10];
            B2_C2:s -> B2_C3:n [weight=10];
            B2_C3:s -> B2_Add:n [weight=10];
            
            // 侧边直连分支
            B2_In:e -> B2_Id:n [weight=1];
            B2_Id:s -> B2_Add:e [weight=1];
            
            B2_Add:s -> B2_Out:n [weight=10];
        }
    }
    
    // 利用一条隐形线，将 Bottleneck 区域推到主网络右侧
    Layer1:e -> B1_In:w [style=invis, minlen=2];
}

# 主干网络 LSTM (RNN) deeplabv3

    digraph LSTM_DeepLabV3_1D {
        // 全局设定：直角连线，紧凑整洁的排版
        graph [rankdir=TB, splines=ortho, nodesep=0.6, ranksep=0.6, fontname="Helvetica"];
        node [shape=box, style="rounded,filled", fillcolor=white, fontname="Helvetica", fontsize=11, margin=0.15];
        edge [fontname="Helvetica", fontsize=10, color="#444444", penwidth=1.2];
    
        // ==========================================
        // 输入节点
        // ==========================================
        Input [label="Input\nShape: (6, L)", fillcolor="#e6f2ff", color="#0066cc", penwidth=2];
    
        // ==========================================
        // LSTM 混合主干网络 (CNN Stem + BiLSTM)
        // ==========================================
        subgraph cluster_backbone {
            label="BiLSTM 1D Backbone (with CNN Stem)";
            style=dashed; color="#888888"; bgcolor="#fafafa";
            
            // 降采样 Stem
            Stem_Conv [label="Conv1d(k=7, s=2, p=3)\nBN, ReLU\nOut: (64, L/2)"];
            Stem_Pool [label="MaxPool1d(k=3, s=2, p=1)\nOut: (64, L/4)"];
            
            // 维度转换适应 RNN
            Permute_In [label="Permute\n(Channels, Length) -> (Length, Channels)", shape=cds, fillcolor="#f0f0f0"];
            
            // LSTM 层
            LSTM_Layer1 [label="BiLSTM Layer 1\nSequence Processing\nOut: (Length, 512)"];
            LSTM_Layer2 [label="BiLSTM Layer 2\nSequence Processing\nOut: (Length, 1024)"];
            LSTM_Layer3 [label="BiLSTM Layer 3\nSequence Processing\nOut: (Length, 2048)"];
            
            // 维度转换回适应 Conv1d
            Permute_Out [label="Permute\n(Length, Channels) -> (Channels, Length)", shape=cds, fillcolor="#f0f0f0"];
        }
    
        // 主干道路由控制 (超级权重保证笔直)
        Input:s -> Stem_Conv:n [weight=100];
        Stem_Conv:s -> Stem_Pool:n [weight=100];
        Stem_Pool:s -> Permute_In:n [weight=100];
        Permute_In:s -> LSTM_Layer1:n [weight=100];
        LSTM_Layer1:s -> LSTM_Layer2:n [weight=100];
        LSTM_Layer2:s -> LSTM_Layer3:n [weight=100];
        LSTM_Layer3:s -> Permute_Out:n [weight=100];
    
        // ==========================================
        // FCN Head 辅助分支
        // ==========================================
        subgraph cluster_fcn {
            label="FCN Head (Aux) 1D";
            style=solid; color="#cc9900"; bgcolor="#fff9e6";
            
            // 提前将维度转回 Conv1d 格式供辅助分支使用
            Permute_Aux [label="Permute\n(L, C) -> (C, L)", shape=cds, fillcolor="#f0f0f0"];
            FCN_Conv1 [label="Conv1d(k=3, s=1, p=1)\nBN, ReLU"];
            FCN_Drop [label="Dropout"];
            FCN_Conv2 [label="Conv1d(k=1, s=1)"];
            FCN_Interp [label="Linear Interpolate\nto length L"];
            Aux_Out [label="Aux Output\n(num_classes, L)", fillcolor="#e6ffe6", color="#00aa00", penwidth=2];
    
            Permute_Aux:s -> FCN_Conv1:n [weight=10];
            FCN_Conv1:s -> FCN_Drop:n [weight=10];
            FCN_Drop:s -> FCN_Conv2:n [weight=10];
            FCN_Conv2:s -> FCN_Interp:n [weight=10];
            FCN_Interp:s -> Aux_Out:n [weight=10];
        }
        
        // 从 LSTM Layer 2 引出辅助分支，模拟原 ResNet Layer 3 的位置
        LSTM_Layer2:e -> Permute_Aux:n [xlabel=" Aux 分支 ", weight=1];
    
        // ==========================================
        // ASPP 模块 (连接在主干最终输出后)
        // ==========================================
        subgraph cluster_aspp {
            label="ASPP 1D";
            style=solid; color="#0066cc"; bgcolor="#eef9ff";
            
            { rank=same; ASPP_B1; ASPP_B2; ASPP_B3; ASPP_B4; ASPP_B5; }
    
            ASPP_B1 [label="Conv1d\n(k=1, s=1)"];
            ASPP_B2 [label="Conv1d\n(k=3, s=1, r=12)"];
            ASPP_B3 [label="Conv1d\n(k=3, s=1, r=24)"];
            ASPP_B4 [label="Conv1d\n(k=3, s=1, r=36)"];
            ASPP_B5 [label="AdaptiveAvgPool1d(1)\nConv1d(k=1, s=1)\nLinear Interp"];
            
            Concat [label="Concat", fillcolor="#ffe6cc", shape=folder];
            ASPP_Out [label="Conv1d(k=1, s=1)\nBN, ReLU"];
            ASPP_Drop [label="Dropout"];
    
            ASPP_B1:s -> Concat:n;
            ASPP_B2:s -> Concat:n;
            ASPP_B3:s -> Concat:n;
            ASPP_B4:s -> Concat:n;
            ASPP_B5:s -> Concat:n;
            
            Concat:s -> ASPP_Out:n [weight=100];
            ASPP_Out:s -> ASPP_Drop:n [weight=100];
        }
    
        // 从主干最终的 Permute_Out 分配给 ASPP
        Permute_Out:s -> ASPP_B1:n;
        Permute_Out:s -> ASPP_B2:n;
        Permute_Out:s -> ASPP_B3:n;
        Permute_Out:s -> ASPP_B4:n;
        Permute_Out:s -> ASPP_B5:n;
    
        // ==========================================
        // DeepLab Main Head 主分支
        // ==========================================
        subgraph cluster_head {
            label="DeepLab Head 1D";
            style=solid; color="#cc9900"; bgcolor="#fff9e6";
            
            Head_Conv1 [label="Conv1d(k=3, s=1, p=1)\nBN, ReLU"];
            Head_Conv2 [label="Conv1d(k=1, s=1)"];
            Head_Interp [label="Linear Interpolate\nto length L"];
            Main_Out [label="Main Output\n(num_classes, L)", fillcolor="#e6ffe6", color="#00aa00", penwidth=2];
    
            Head_Conv1:s -> Head_Conv2:n [weight=10];
            Head_Conv2:s -> Head_Interp:n [weight=10];
            Head_Interp:s -> Main_Out:n [weight=10];
        }
        
        ASPP_Drop:s -> Head_Conv1:n [weight=100];
    }
}



### 1. 边缘端硬件加速的“水土不服”与替代方案 (TCN)

在将此类监控模型转化为实际工业应用时，尤其是准备进行边缘端 NPU（例如瑞芯微 RK3576/RK3588 等芯片）的交叉编译和部署时，硬件的算力特性对网络结构有决定性的反作用。

- **痛点**：目前绝大多数 NPU 对纯卷积（CNN）架构的算子支持和加速效率极高，但对 RNN/LSTM 这类具有时间状态依赖、难以高度并行化的网络支持较弱。强行部署 LSTM 可能会导致 NPU 利用率低下，甚至回退到 CPU 计算，使得推理延迟大幅增加。
- **建议架构**：可以考察 **TCN (时间卷积网络, Temporal Convolutional Network)**。TCN 使用一维因果膨胀卷积（1D Causal Dilated Convolution），既具备超越 LSTM 的长感受野和时序记忆能力，又能完美转化为标准 Conv1d 算子，在 NPU 上跑出极高的并发效率。

### 2. ASPP 膨胀率 (Dilation Rates) 的重新标定

DeepLabV3+ 原生的 ASPP 膨胀率 $(r=12, 24, 36)$ 是为大分辨率二维图像（如自动驾驶街景）设计的。

- **痛点**：在一维多通道信号中，经过 Stem 层的降采样后，序列长度 $L$ 可能已经缩短。如果盲目使用如此大的膨胀率，卷积核的采样点之间会跨越过大，提取到的特征不仅稀疏，还容易引发“网格效应”（Gridding Effect），丢失连续的振动或电流特征。
- **建议架构**：根据你实际输入序列的物理时间长度和降采样倍率，重新设计一组**互质**的膨胀率。例如，改为 $(r=2, 5, 9)$ 或 $(r=3, 7, 11)$。互质的膨胀率可以保证卷积核在不同层级间能够“填满”所有的采样间隙，让时域特征更加连贯。

### 3. 多源数据异构融合 (Multi-branch Input)

你的输入是 6 通道的一维数据。在刀具磨损等工业监测中，这 6 个通道往往来自不同类型的传感器（比如：X/Y/Z轴的高频振动信号、主轴的低频电流/电压信号）。

- **痛点**：直接将不同物理量、不同采样率特性的信号在通道维度拼接后输入同一个 $k=7$ 的卷积核，会让网络在前期的特征提取非常吃力。
- **建议架构**：在网络的最前端采用**分组多分支 Stem 结构**。
    - **高频分支**：针对振动信号通道，使用小卷积核（如 $k=3$ 或 $k=5$）来捕捉瞬态冲击特征。
    - **低频分支**：针对电流/温度信号通道，使用大卷积核（如 $k=15$ 或 $k=31$）或较大的步长来提取宏观趋势。
    - **融合机制**：提取完毕后再进行 Concat，并接一个 $1 \times 1$ 的 Conv1d 进行跨通道信息交互，再送入主干网络。

### 4. 引入一维注意力机制 (1D Attention)

刀具在不同切削阶段，对不同传感器通道的敏感度是动态变化的。

- **建议架构**：在 CNN Stem 之后、主干网络之前，或者主干输出与 ASPP 之间，插入一维的 **CBAM (卷积块注意力模型)** 或 **SE (Squeeze-and-Excitation)** 模块。这只需要极其微小的计算开销，就能让网络学会“动态静音”当前无关的噪声通道，并对关键特征通道（如突然异常的 Z 轴振动）进行权重放大。
- 

综上所述，如果为了追求极致的时序捕捉，可以继续深挖 LSTM；但如果最终目标是要在嵌入式板卡上实现高频、实时的工况量化推理，回归到优化过的一维全卷积架构（TCN + 改进的 ASPP）会是工程上阻力最小的路径。