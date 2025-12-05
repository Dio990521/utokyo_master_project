import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import pandas as pd
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.drawing_env.tools.custom_cnn import CustomCnnExtractor


class TrainingDataCallback(BaseCallback):
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path
        self.episode_data = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if "used_budgets" in info:
                    self.episode_data.append({
                        "pixel_similarity": info.get("pixel_similarity"),
                        "recall_black": info.get("recall_black"),
                        "recall_grey": info.get("recall_grey"),
                        "recall_all": info.get("recall_all"),
                        "recall_white": info.get("recall_white"),
                        "used_budgets": info.get("used_budgets"),
                        "total_painted": info.get("total_painted"),
                        "correctly_painted": info.get("correctly_painted"),
                        "precision": info.get("precision"),
                        "f1_score": info.get("f1_score"),
                        "episode_combo_log": info.get("episode_combo_log"),
                        "episode_base_reward": info.get("episode_base_reward"),
                        "episode_combo_bonus": info.get("episode_combo_bonus"),
                        "combo_sustained": info.get("combo_sustained"),
                    })
                    self.logger.record("precision", info.get("precision"))
                    self.logger.record("recall_black", info.get("recall_black"))
                    self.logger.record("recall_grey", info.get("recall_grey"))
                    self.logger.record("recall_all", info.get("recall_all"))
                    self.logger.record("f1_score", info.get("f1_score"))
                    self.logger.record("total_painted", info.get("total_painted"))
                    self.logger.record("correctly_painted", info.get("correctly_painted"))
        return True

    def _on_training_end(self) -> None:
        if not self.episode_data:
            print("[Callback] No training episode data collected.")
            return
        df = pd.DataFrame(self.episode_data)
        df.to_csv(self.save_path, index_label="episode")
        print(f"[Callback] Saved training data to {self.save_path}")


class ValidationCallback(BaseCallback):
    # (Keep validation logic as is, it depends on env which is already updated)
    def __init__(self, eval_env_config: dict, eval_freq: int, save_path: str):
        super().__init__()
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.eval_env_config = eval_env_config
        self.validation_data = []
        val_path = self.eval_env_config["val_sketches_path"]
        self.val_sketch_files = [os.path.join(val_path, f) for f in os.listdir(val_path) if
                                 f.endswith(('.png', '.jpg'))]
        if not self.val_sketch_files: raise ValueError(f"No validation sketches in {val_path}!")

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            print(f"\n--- Validation at step {self.num_timesteps} ---")
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
                    "pixel_similarity": info.get("pixel_similarity"),
                    "recall_black": info.get("recall_black"),
                    "recall_grey": info.get("recall_grey"),
                    "recall_all": info.get("recall_all"),
                    "recall_white": info.get("recall_white"),
                    "used_budgets": info.get("used_budgets"),
                    "total_painted": info.get("total_painted"),
                    "correctly_painted": info.get("correctly_painted"),
                    "precision": info.get("precision"),
                    "f1_score": info.get("f1_score"),
                    "episode_combo_log": info.get("episode_combo_log"),
                    "episode_base_reward": info.get("episode_base_reward"),
                    "episode_combo_bonus": info.get("episode_combo_bonus"),
                    "combo_sustained": info.get("combo_sustained"),
                })
                eval_env.close()
            self.validation_data.extend(results)
            avg_sim = np.mean([res["pixel_similarity"] for res in results])
            self.logger.record("validation/avg_pixel_similarity", avg_sim)
            self.logger.dump(self.num_timesteps)
            print(f"--- Validation Complete. Avg Similarity: {avg_sim:.4f} ---")
        return True

    def _on_training_end(self) -> None:
        if self.validation_data:
            pd.DataFrame(self.validation_data).to_csv(self.save_path, index=False)
            print(f"[Validation] Saved data to {self.save_path}")


def run_training(config: dict):
    VERSION = config.get("VERSION", "_default_run")
    TOTAL_TIME_STEPS = config.get("TOTAL_TIME_STEPS", 5000000)
    LEARNING_RATE = config.get("LEARNING_RATE", 0.0003)
    ENT_COEF = config.get("ENT_COEF", 0.01)
    NUM_ENVS = config.get("NUM_ENVS", 1)
    BATCH_BASE_SIZE = config.get("BATCH_SIZE", 32)
    SEED = config.get("SEED", None)

    env_config = config.get("ENV_CONFIG", {})
    cnn_padding = env_config.get("cnn_padding", True)
    validation_config = config.get("VALIDATION_CONFIG", None)
    ENV_ID = config.get("ENV_ID", "DrawingEnv-v0")
    print(f"Training on ENV_ID: {ENV_ID}")
    BASE_OUTPUT_DIR = f"../training_outputs/{VERSION}/"
    LOG_DIR = os.path.join(BASE_OUTPUT_DIR, "logs/")
    MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models/")
    TRAINING_DATA_PATH = os.path.join(BASE_OUTPUT_DIR, "training_data.csv")
    VALIDATION_DATA_PATH = os.path.join(BASE_OUTPUT_DIR, "validation_data.csv")
    model_path = os.path.join(MODELS_DIR, "drawing_agent_final.zip")

    STEP_DEBUG_DIR = os.path.join(BASE_OUTPUT_DIR, "step_debug/")
    env_config["step_debug_path"] = STEP_DEBUG_DIR

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(STEP_DEBUG_DIR, exist_ok=True)

    env = make_vec_env(
        ENV_ID,
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": env_config}
    )

    # Standard CNN Policy
    print("[Training] Using CnnPolicy (Standard)")
    policy_type = "CnnPolicy"
    policy_kwargs = dict(
        features_extractor_class=CustomCnnExtractor,
        features_extractor_kwargs=dict(features_dim=128, padding=cnn_padding),
    )

    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}. Loading...")
        model = PPO.load(model_path, env=env, tensorboard_log=LOG_DIR)
    else:
        print(f"Creating new model with {policy_type}...")
        model = PPO(
            policy_type,
            env,
            learning_rate=LEARNING_RATE,
            n_steps=2048,
            batch_size=BATCH_BASE_SIZE,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ENT_COEF,
            verbose=1,
            tensorboard_log=LOG_DIR,
            policy_kwargs=policy_kwargs,
            seed=SEED,
        )

    config["SEED"] = model.seed
    callbacks = [TrainingDataCallback(save_path=TRAINING_DATA_PATH)]
    if validation_config:
        callbacks.append(ValidationCallback(
            eval_env_config=validation_config["ENV_CONFIG"],
            eval_freq=validation_config["EVAL_FREQ"],
            save_path=VALIDATION_DATA_PATH
        ))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=callbacks, reset_num_timesteps=False)

    save_name = "drawing_agent_final.zip" if not os.path.exists(model_path) else "drawing_agent_final_new.zip"
    model.save(os.path.join(MODELS_DIR, save_name))
    print(f"Model saved to {MODELS_DIR}")
    env.close()