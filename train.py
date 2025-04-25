import gymnasium as gym
import envs.mouse_click

env = gym.make('MouseClick-v0')
print(gym.envs.registry['MouseClick-v0'])
env.reset()