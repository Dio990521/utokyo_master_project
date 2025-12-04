import cv2
import numpy as np
from gymnasium.utils.env_checker import check_env
import gymnasium as gym

from envs.drawing_env.draw_env_grey import DrawingAgentGreyEnv

env = gym.make("DrawingEnv-v0")

check_env(env)