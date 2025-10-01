# train/train.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import pandas as pd
import numpy as np

from envs.drawing_env.tools.custom_cnn import CustomCnnExtractor


class TrainingDataCallback(BaseCallback):
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path
        self.episode_data = []

    def _on_step(self) -> bool:
        if self.locals["dones"]:
            info = self.locals["infos"][0]
            if "similarity" in info:
                self.episode_data.append({
                    "similarity": info.get("similarity"),
                    "used_budgets": info.get("used_budgets"),
                    "block_similarity": info.get("block_similarity"),
                    "block_reward": info.get("block_reward"),
                    "step_rewards": info.get("step_rewards"),
                })
        return True

    def _on_training_end(self) -> None:
        if not self.episode_data:
            print("[Callback] No training episode data was collected.")
            return
        df = pd.DataFrame(self.episode_data)
        df.to_csv(self.save_path, index_label="episode")
        print(f"[Callback] Successfully saved training data to {self.save_path}")


class ValidationCallback(BaseCallback):
    def __init__(self, eval_env_config: dict, eval_freq: int, save_path: str):
        super().__init__()
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.eval_env_config = eval_env_config
        self.validation_data = []

        val_path = self.eval_env_config["target_sketches_path"]
        self.val_sketch_files = [os.path.join(val_path, f) for f in os.listdir(val_path) if
                                 f.endswith(('.png', '.jpg'))]
        if not self.val_sketch_files:
            raise ValueError(f"No validation sketches found in {val_path}!")
        print(f"[Validation] Found {len(self.val_sketch_files)} sketches for validation.")

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            print(f"\n--- Running Validation at step {self.num_timesteps} ---")

            results = []
            for sketch_file in self.val_sketch_files:
                temp_config = {**self.eval_env_config, "specific_sketch_file": sketch_file}
                eval_env = gym.make("DrawingEnv-v0", config=temp_config)

                obs, _ = eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, _, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated

                results.append({
                    "step": self.num_timesteps,
                    "sketch": os.path.basename(sketch_file),
                    "similarity": info.get("similarity"),
                    "used_budgets": info.get("used_budgets"),
                })
                eval_env.close()

            self.validation_data.extend(results)
            avg_sim = np.mean([res["similarity"] for res in results])
            self.logger.record("validation/avg_similarity", avg_sim)
            self.logger.dump(self.num_timesteps)
            print(f"--- Validation Complete. Average Similarity: {avg_sim:.4f} ---")
        return True

    def _on_training_end(self) -> None:
        if not self.validation_data:
            print("[Validation] No validation data collected.")
            return
        pd.DataFrame(self.validation_data).to_csv(self.save_path, index=False)
        print(f"[Validation] Saved validation data to {self.save_path}")


def run_training(config: dict):
    VERSION = config.get("VERSION", "_default_run")
    TOTAL_TIME_STEPS = config.get("TOTAL_TIME_STEPS", 5000000)
    LEARNING_RATE = config.get("LEARNING_RATE", 0.0003)
    ENT_COEF = config.get("ENT_COEF", 0.01)

    env_config = config.get("ENV_CONFIG", {})
    validation_config = config.get("VALIDATION_CONFIG", None)

    BASE_OUTPUT_DIR = f"../training_outputs/{VERSION}/"
    LOG_DIR = os.path.join(BASE_OUTPUT_DIR, "logs/")
    MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models/")
    TRAINING_DATA_PATH = os.path.join(BASE_OUTPUT_DIR, "training_data.csv")
    VALIDATION_DATA_PATH = os.path.join(BASE_OUTPUT_DIR, "validation_data.csv")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    env = gym.make("DrawingEnv-v0", config=env_config)

    policy_kwargs = dict(features_extractor_class=CustomCnnExtractor, features_extractor_kwargs=dict(features_dim=128))

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs,
    )

    # --- 核心修改：设置回调函数列表 ---
    callbacks = [TrainingDataCallback(save_path=TRAINING_DATA_PATH)]
    if validation_config:
        callbacks.append(ValidationCallback(
            eval_env_config=validation_config["ENV_CONFIG"],
            eval_freq=validation_config["EVAL_FREQ"],
            save_path=VALIDATION_DATA_PATH
        ))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=callbacks)

    model.save(os.path.join(MODELS_DIR, "drawing_agent_final.zip"))
    print(f"Model for {VERSION} saved to {MODELS_DIR}")
    env.close()