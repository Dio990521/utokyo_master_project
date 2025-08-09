import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import pandas as pd

from envs.drawing_env.tools.custom_cnn import CustomCnnExtractor

class TrainingDataCallback(BaseCallback):
    def __init__(self, save_path: str, delta_save_path=None):
        super().__init__()
        self.save_path = save_path
        self.episode_data = []
        self.all_delta_similarities = []
        self.delta_save_path = delta_save_path


    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            info = self.locals.get("infos", [{}])[0]
            data_row = {
                "similarity": info.get("similarity"),
                "used_budgets": info.get("used_budgets"),
                "block_similarity": info.get("block_similarity"),
                "block_reward": info.get("block_reward"),
            }
            self.episode_data.append(data_row)
            if "delta_similarity_history" in info:
                self.all_delta_similarities.extend(info["delta_similarity_history"])
        return True

    def _on_training_end(self) -> None:
        print("\nTraining ended. Saving collected training data...")
        if not self.episode_data:
            print("[Callback] No episode data was collected. Nothing to save.")
            return

        df = pd.DataFrame(self.episode_data)

        try:
            df.to_csv(self.save_path, index_label="episode")
            print(f"[Callback] Successfully saved training data to {self.save_path}")
        except Exception as e:
            print(f"[Callback] Error saving data: {e}")

        print("\nSaving step-level delta similarity data...")
        if not self.all_delta_similarities:
            print("[Callback] No delta similarity data was collected.")
            return

        delta_df = pd.DataFrame(self.all_delta_similarities, columns=['delta_similarity'])
        try:
            delta_df.to_csv(self.delta_save_path, index=False)
            print(f"[Callback] Successfully saved delta similarity data to {self.delta_save_path}")
        except Exception as e:
            print(f"[Callback] Error saving delta similarity data: {e}")


VERSION = "_1"
BASE_OUTPUT_DIR = f"../training_outputs/{VERSION}/"
LOG_DIR = os.path.join(BASE_OUTPUT_DIR, "logs/")
MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models/")
EPISODE_DATA_PATH = os.path.join(BASE_OUTPUT_DIR, "episode_data.csv")
DELTA_DATA_PATH = os.path.join(BASE_OUTPUT_DIR, "delta_similarity_data.csv")

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

training_callback = TrainingDataCallback(
    save_path=EPISODE_DATA_PATH,
    delta_save_path = DELTA_DATA_PATH
)

print("Start training...")
model.learn(
    total_timesteps=TOTAL_TIME_STEPS,
    callback=[training_callback],
)
print("Training finished.")

model.save(os.path.join(MODELS_DIR, "drawing_agent_final.zip"))
print("Model saved to models/drawing_agent_final.zip")