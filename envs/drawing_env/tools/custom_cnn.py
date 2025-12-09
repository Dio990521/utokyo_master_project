import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCordCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        original_channels = observation_space.shape[0]
        n_input_channels = original_channels + 2

        self.cnn = nn.Sequential(
            # 32x32 -> 32x32
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 16x16 -> 8x8
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            n_flatten = 64 * 8 * 8

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = observations.shape
        device = observations.device

        y_coords = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).repeat(batch_size, 1, 1, w)
        x_coords = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).repeat(batch_size, 1, h, 1)
        obs_with_coords = torch.cat([observations, x_coords, y_coords], dim=1)

        return self.linear(self.cnn(obs_with_coords))

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
