import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

from envs.drawing_env.tools.custom_cnn import CustomCnnExtractor


class TensorboardCallbackDraw(BaseCallback):

    def __init__(self, env, save_file_name="test"):
        super().__init__()
        self.env = env
        self.shortest_steps_per_episode = []
        self.ratio_success = []
        self.similarity = []
        self.used_budgets = []
        self.save_file_name = save_file_name + ".csv"

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        for key in info:
            if key == "episode_end":
                if info[key]:
                    self.similarity.append(info["similarity"])
                    self.used_budgets.append(info["used_budgets"])
            else:
                self.logger.record(str(key), info[key])
        return True

    def _on_training_end(self) -> None:
        with open(self.save_file_name, "w") as f:
            for value in self.similarity:
                f.write(f"{value}\n")
        with open("used_budgets" + self.save_file_name, "w") as f:
            for value in self.used_budgets:
                f.write(f"{value}\n")
        print(f"[Callback] Saved")

VERSION = "_test_scale_rb_3"
LOG_DIR = "../envs/drawing_env/training/saved_logs/" + VERSION + "/"
MODELS_DIR = "../envs/drawing_env/training/saved_models/" + VERSION + "/"
SAVE_FILE_NAME = "similarity" + VERSION
MAX_EPISODE_STEPS = 1000
TOTAL_TIME_STEPS = 5000000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

env = gym.make("DrawingEnv-v0")

policy_kwargs = dict(
    features_extractor_class=CustomCnnExtractor,
    features_extractor_kwargs=dict(features_dim=128)
)

model = PPO(
    "CnnPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=LOG_DIR,
    policy_kwargs=policy_kwargs,
)

# checkpoint_callback = CheckpointCallback(
#     save_freq=MAX_EPISODE_STEPS * 100,
#     save_path=MODELS_DIR,
#     name_prefix="drawing_agent_checkpoint",
#     save_replay_buffer=True,
#     save_vecnormalize=True,
# )


print("Start training...")
model.learn(
    total_timesteps=TOTAL_TIME_STEPS,
    callback=[TensorboardCallbackDraw(env, save_file_name=SAVE_FILE_NAME)],
)
print("Training finished.")

model.save(os.path.join(MODELS_DIR, "drawing_agent_final.zip"))
print("Model saved to models/drawing_agent_final.zip")