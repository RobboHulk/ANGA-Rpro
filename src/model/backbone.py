"""
================================================================================
备用骨干网络：ResNet 与 TextEncoder
================================================================================
说明：本文件实现了 ResNet（图像/音频骨干）与基于 BERT 的 TextEncoder
（文本骨干），但【当前 ANGA 主模型使用的是 ViLT 作为骨干】，
并未实际调用本文件中的网络（在 model/__init__.py 中也没有导出）。

保留本文件的原因推测是：作者早期实验阶段曾尝试使用"ResNet + BERT"双流
结构（two-stream）作为多模态骨干，后期改用 ViLT 单流结构后此文件作为
历史代码留存，供后续扩展或对照实验使用。

内容概览：
    - conv3x3 / conv1x1      : 卷积辅助函数
    - BasicBlock / Bottleneck: ResNet 的基础残差块
    - ResNet                 : 支持 visual/audio 两种模态的 ResNet 主干
    - resnet18 / resnet50    : 常用 ResNet 变体工厂函数
    - TextEncoder            : 封装 BERT 的文本编码器
================================================================================
"""
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # 3x3 卷积（padding 与 dilation 相同以保证输出尺寸不变）
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # 1x1 卷积（用于改变通道数或降采样）
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    ResNet-18/34 使用的基础残差块：
    结构为 [3x3 conv → BN → ReLU → 3x3 conv → BN] + 残差连接。
    """
    expansion = 1  # 该块的输出通道数是输入通道数的 1 倍

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # BasicBlock 仅支持 groups=1 与 base_width=64（与官方 ResNet 一致）
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # 当 stride != 1 时，conv1 与 downsample 都会对输入做下采样
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample  # 残差分支的降采样层（若需要）
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存输入作为残差连接的恒等映射

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 若需要降采样，则对恒等分支做相应变换，使两者尺寸匹配
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity   # 残差相加
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 主干网络。支持 visual（3 通道图像）与 audio（1 通道音频）两种输入。
    """

    def __init__(self, block, layers, modality, num_classes=1000, pool='avgpool', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.modality = modality   # 'visual' 或 'audio'
        self.pool = pool
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64         # 第一层卷积后的通道数
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 每个元素表示是否把该 stage 的 2x2 stride 替换为膨胀卷积
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # ---- 首层：7x7 卷积（stride=2）----
        if modality == 'audio':
            # 音频：单通道输入
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'visual':
            # 图像：RGB 三通道输入
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            raise NotImplementedError('Incorrect modality, should be audio or visual but got {}'.format(modality))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- 四个残差 stage ----
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # （以下分类池化/全连接层在早期方案中被注释，当前 ResNet 只输出特征图）
        # if self.pool == 'avgpool':
        #     self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)  # 8192

        # ---- 权重初始化 ----
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # （可选）把每个残差分支最后一个 BN 的权重初始化为 0，使残差块初始等价于恒等映射，
        # 可带来约 0.2~0.3% 的提升（见 ResNet v1.5 论文）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """构建一个残差 stage（包含 blocks 个残差块）。"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 当 stride != 1 或通道数不匹配时，需要给恒等分支加一个降采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 第一个块可能带降采样
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # 后续块不再降采样
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 若输入是视觉模态的 5D 张量 (B, C, T, H, W)，即带时间维的帧序列，
        # 先重排为 (B, T, C, H, W) 再合并 B*T 维，以便逐帧处理
        if self.modality == 'visual':
            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B * T, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = x   # 输出特征图（未接全局池化与全连接层）

        return out


class Bottleneck(nn.Module):
    """
    ResNet-50/101/152 使用的瓶颈残差块：
    结构为 [1x1 降维 → 3x3 卷积 → 1x1 升维] + 残差连接。
    """
    expansion = 4  # 输出通道数是中间通道数的 4 倍

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  # 中间层宽度
        # conv2 与 downsample 在 stride != 1 时都会下采样
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _resnet(arch, block, layers, modality, progress, **kwargs):
    """ResNet 工厂函数：按参数构造 ResNet 实例。"""
    print(layers)
    model = ResNet(block, layers, modality, **kwargs)
    # （以下是早期加载 ImageNet 预训练权重的代码，已注释）
    # if modality == 'audio':
    #     print('use modality:', modality)
    #     pretrained_dict = torch.load('/data/gzh/MMNAS/two_stage_resnet/logs/resnet18-5c106cde.pth')
    #     pretrained_dict['conv1.weight'] = pretrained_dict['conv1.weight'].mean(dim=1, keepdim=True)
    #     print(model.load_state_dict(pretrained_dict, strict=False))
    #     # for param in model.parameters():
    #     #     param.requires_grad = False
    # elif modality == 'visual':
    #     print('use modality:', modality)
    #     pretrained_dict = torch.load('/data/gzh/MMNAS/two_stage_resnet/logs/resnet18-5c106cde.pth')
    #     print(model.load_state_dict(pretrained_dict, strict=False))
    #     # for param in model.parameters():
    #     #     param.requires_grad = False
    # else:
    #     print('no pretrain weights!')

    return model

def resnet18(modality, progress=True, **kwargs):
    """构造 ResNet-18（每 stage 2 个 BasicBlock）。"""
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], modality, progress, **kwargs)

def resnet50(modality, progress=True, **kwargs):
    """构造 ResNet-50（每 stage [3,4,6,3] 个 Bottleneck）。"""
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], modality, progress, **kwargs)


from transformers import BertModel, BertTokenizer
class TextEncoder(nn.Module):
    """
    基于 BERT 的文本编码器（早期双流方案的文本分支）。
    """
    def __init__(self, bert_path, fine_tune=True):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)   # BERT 分词器
        self.bert = BertModel.from_pretrained(bert_path)            # BERT 模型
        self.fine_tune(fine_tune)                                   # 决定是否微调

    def fine_tune(self, fine_tune=True):
        # 如果不进行微调，则冻结 BERT 模型的所有参数
        if not fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

    def re_text_input(self, texts):
        # 将原始文本编码为 input_ids/attention_mask 等（最长 padding，最大 30 长度）
        return self.tokenizer(texts, padding='longest', max_length=30, return_tensors="pt")

    def forward(self, texts):
        input_ids = texts['input_ids']
        attention_masks = texts['attention_mask']
        outputs = self.bert(input_ids, attention_mask=attention_masks)
        # 返回最后一层隐藏状态与 pooler 输出（[CLS] 表征）
        return outputs.last_hidden_state, outputs.pooler_output