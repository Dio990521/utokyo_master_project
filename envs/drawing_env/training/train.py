import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.drawing_env.draw_env import DrawingAgentEnv
from envs.drawing_env.tools.image_process import ImageDraw
import envs.drawing_env
import os

VERSION = "_1/"
LOG_DIR = "saved_logs/" + VERSION
MODELS_DIR = "saved_models/" + VERSION
SKETCH_DATA_PATH = "sketches/"
CANVAS_SIZE = (100, 100)
MAX_EPISODE_STEPS = 1000
TOTAL_TIME_STEPS = 1000000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SKETCH_DATA_PATH, exist_ok=True)

env = gym.make("DrawingEnv-v0")
keys_to_normalize = ["canvas", "target_sketch", "cursor_pos"]

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    # features_extractor_class=CustomCNNFeaturesExtractor
)

model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.1,
    verbose=1,
    tensorboard_log=LOG_DIR,
    policy_kwargs=policy_kwargs,
)

checkpoint_callback = CheckpointCallback(
    save_freq=MAX_EPISODE_STEPS * 100,
    save_path=MODELS_DIR,
    name_prefix="drawing_agent_checkpoint",
    save_replay_buffer=True,
    save_vecnormalize=True,
)


print("Start training...")
model.learn(
    total_timesteps=TOTAL_TIME_STEPS,
    callback=[checkpoint_callback],
    progress_bar=True
)
print("Training finished.")

model.save(os.path.join(MODELS_DIR, "drawing_agent_final.zip"))
print("Model saved to models/drawing_agent_final.zip")