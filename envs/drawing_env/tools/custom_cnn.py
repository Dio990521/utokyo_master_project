import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCnnExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
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


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_output_dim: int = 128, mlp_output_dim: int = 32):

        total_features_dim = cnn_output_dim + mlp_output_dim
        super().__init__(observation_space, features_dim=total_features_dim)

        image_space = observation_space["image"]
        n_input_channels = image_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            sample_image = torch.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.cnn(sample_image).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU()
        )

        state_space = observation_space["state"]
        state_dim = state_space.shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, mlp_output_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        image_input = observations["image"]
        if image_input.shape[1] > 0:
            cnn_features = self.cnn(image_input)
            cnn_features = self.cnn_linear(cnn_features)
        else:
            cnn_features = torch.zeros(image_input.shape[0], self.cnn_linear[0].out_features, device=image_input.device)

        state_features = self.mlp(observations["state"])

        return torch.cat([cnn_features, state_features], dim=1)