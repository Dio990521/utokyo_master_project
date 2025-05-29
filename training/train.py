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

    def __init__(self, env, save_file_name, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.clicked_targets_per_episode = []
        self.save_file_name = save_file_name + ".csv"

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        if "click_times" in info:
            if info["episode_end"]:
                self.clicked_targets_per_episode.append(info["clicked_targets"])
            self.logger.record("click frequency", info["click_times"])
            self.logger.record("distance reward", info["distance_reward"])
            self.logger.record("cosine similarity reward", info["cossim_reward"])
            self.logger.record("clicked targets", info["clicked_targets"])
        return True

    def _on_training_end(self) -> None:
        with open(self.save_file_name, "w") as f:
            for value in self.clicked_targets_per_episode:
                f.write(f"{value}\n")
            print(f"[Callback] Saved clicked_targets to {self.save_file_name}")

env = gym.make('MouseClick-v0')
print(gym.envs.registry['MouseClick-v0'])

env.reset()
env.render()
log_name = "simple_action_obs_no4"
#model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=f"./ppo_tensorboard_{log_name}/", ent_coef=0.1)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./ppo_tensorboard_{log_name}/", ent_coef=0.1)

model.learn(total_timesteps=1000000, callback=TensorboardCallback(env, log_name))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = "saved_agents/" + log_name
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, f"ppo_click_env_{timestamp}"))
