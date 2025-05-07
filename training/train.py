import os
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('MouseClick-v0')
print(gym.envs.registry['MouseClick-v0'])

env.reset()
env.render()

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

model.learn(total_timesteps=1000000)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = "saved_agents"
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, f"ppo_click_env_{timestamp}"))
model.save(f"ppo_click_env_{timestamp}")