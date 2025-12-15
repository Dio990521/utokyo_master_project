import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCnnExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Space, features_dim: int = 128, padding=True):
        super().__init__(observation_space, features_dim)

        self.use_dist_val = isinstance(observation_space, spaces.Dict)

        if self.use_dist_val:
            image_space = observation_space.spaces["image"]
            dist_val = observation_space.spaces["dist_val"]

            n_input_channels = image_space.shape[0]
            dist_val_dim = dist_val.shape[0]
        else:
            image_space = observation_space
            n_input_channels = image_space.shape[0]
            dist_val_dim = 0

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
            sample_input = torch.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + dist_val_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        if self.use_dist_val:
            img_tensor = observations["image"]
            dist_val = observations["dist_val"]
            cnn_features = self.cnn(img_tensor)
            combined_features = torch.cat((cnn_features, dist_val), dim=1)
            return self.linear(combined_features)
        return self.linear(self.cnn(observations))