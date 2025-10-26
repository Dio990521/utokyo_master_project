import cv2
import numpy as np
from gymnasium.utils.env_checker import check_env
import gymnasium as gym

from envs.drawing_env.draw_env import DrawingAgentEnv

env = gym.make("DrawingEnv-v0")

check_env(env)