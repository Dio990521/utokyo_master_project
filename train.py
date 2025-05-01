import gymnasium as gym
import envs.mouse_click
from stable_baselines3 import PPO

env = gym.make('MouseClick-v0')
print(gym.envs.registry['MouseClick-v0'])
env.reset()
env.render()

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

model.learn(total_timesteps=1000000)

model.save("ppo_click_env")