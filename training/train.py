import os
from datetime import datetime
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import envs.mouse_click
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if "click_times" in info:
            self.logger.record("click frequency", info["click_times"])
            self.logger.record("distance reward", info["distance_reward"])
            self.logger.record("cosine similarity reward", info["cossim_reward"])
        return True


env = gym.make('MouseClick-v0')
print(gym.envs.registry['MouseClick-v0'])

env.reset()
env.render()

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard_2/", ent_coef=0.1)

model.learn(total_timesteps=2000000, callback=TensorboardCallback(env))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = "saved_agents"
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, f"ppo_click_env_{timestamp}"))
