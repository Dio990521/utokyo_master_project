import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpatialAttention(nn.Module):
    """
    简单的空间注意力模块
    原理：通过对通道维度的最大池化和平均池化，提取空间特征，
    然后通过卷积层学习生成一个 2D 的权重图 (Attention Map)。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 输入通道为 2 (一个是 MaxPool 的结果，一个是 AvgPool 的结果)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: [batch, channels, height, width]

        # 1. 压缩通道信息：平均池化 + 最大池化
        # [b, 1, h, w]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 2. 拼接
        # [b, 2, h, w]
        x_cat = torch.cat([avg_out, max_out], dim=1)

        # 3. 卷积 + Sigmoid 激活 -> 生成注意力图 (0~1 之间)
        # [b, 1, h, w]
        attention_map = self.sigmoid(self.conv1(x_cat))

        # 4. 将注意力图乘回原特征图 (广播机制)
        # 重要的位置数值变大，不重要的背景数值变小
        return x * attention_map


class CustomCnnExtractor2(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, padding=True):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        # 定义空间注意力模块
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # 拆分 CNN 结构，以便在中间插入注意力
        # 建议在第一层卷积后立刻加 Attention，让网络尽早学会“看哪里”
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=int(padding)),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=int(padding)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(padding)),
            nn.ReLU(),
            nn.Flatten()
        )

        # 计算扁平化后的大小
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()

            # 模拟前向传播
            x = self.conv_block1(sample_input)
            x = self.spatial_attn(x)  # 注意力层
            n_flatten = self.conv_block2(x).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. 提取初步特征
        x = self.conv_block1(observations)

        # 2. 应用注意力机制：聚焦重要像素
        x = self.spatial_attn(x)

        # 3. 继续提取高级特征并扁平化
        x = self.conv_block2(x)

        # 4. 全连接层输出
        output = self.linear(x)
        return output

class CustomCnnExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128, padding=True):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=int(padding)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=int(padding)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=int(padding)),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        output = self.linear(self.cnn(observations))
        return output
