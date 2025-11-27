import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_output_dim: int = 128):
        vec_dim = observation_space["vector"].shape[0]
        features_dim = cnn_output_dim + vec_dim

        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space["image"].shape[0]

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
            sample_img = torch.as_tensor(observation_space["image"].sample()[None]).float()
            n_flatten = self.cnn(sample_img).shape[1]

        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        img_obs = observations["image"]
        vec_obs = observations["vector"]

        cnn_out = self.cnn(img_obs)
        cnn_out = self.cnn_linear(cnn_out)  # (Batch, 128)

        combined = torch.cat([cnn_out, vec_obs], dim=1)

        return combined

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
