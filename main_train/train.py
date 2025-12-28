import os
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.drawing_env.tools.custom_cnn import CustomCnnExtractor


# ==========================================
# 1. CALLBACKS
# ==========================================
class TrainingDataCallback(BaseCallback):
    """
    Callback to log detailed training metrics (rewards, combo stats, pixel accuracy)
    at the end of each episode and save them to a CSV file upon training completion.
    """

    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path
        self.episode_data = []

    def _on_step(self) -> bool:
        # Check if any environment in the vectorized env is done
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]

                # If budget info is present, record all relevant metrics
                if "used_budgets" in info:
                    self.episode_data.append({
                        "total_steps": self.num_timesteps,
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
                        "negative_reward": info.get("negative_reward"),
                        "jump_count": info.get("jump_count", 0),
                        "jump_draw_combo_count": info.get("jump_draw_combo_count", 0),
                        "episode_return": info.get("episode_return"),
                        "target_pixel_count": info.get("target_pixel_count", 0),
                    })

                    # Log key metrics to TensorBoard
                    self.logger.record("precision", info.get("precision"))
                    self.logger.record("recall_black", info.get("recall_black"))
                    self.logger.record("recall_grey", info.get("recall_grey"))
                    self.logger.record("recall_all", info.get("recall_all"))
                    self.logger.record("f1_score", info.get("f1_score"))
                    self.logger.record("total_painted", info.get("total_painted"))
                    self.logger.record("correctly_painted", info.get("correctly_painted"))
                    self.logger.record("jump_count", info.get("jump_count", 0))
                    self.logger.record("jump_draw_combo_count", info.get("jump_draw_combo_count", 0))
        return True

    def _on_training_end(self) -> None:
        """Saves accumulated episode data to CSV when training finishes."""
        if not self.episode_data:
            return
        df = pd.DataFrame(self.episode_data)

        # Append to existing file or create new one
        file_exists = os.path.isfile(self.save_path)
        df.to_csv(self.save_path, mode='a', index_label="episode", header=not file_exists)
        print(f"[Callback] Appended training data to {self.save_path}")


class ValidationCallback(BaseCallback):
    """
    Callback to run evaluation episodes on specific validation sketches
    at set intervals (eval_freq) and save the results.
    """

    def __init__(self, eval_env_config: dict, eval_freq: int, save_path: str):
        super().__init__()
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.eval_env_config = eval_env_config
        self.validation_data = []

        # Load validation sketches
        val_path = self.eval_env_config["val_sketches_path"]
        self.val_sketch_files = [os.path.join(val_path, f) for f in os.listdir(val_path) if
                                 f.endswith(('.png', '.jpg'))]
        if not self.val_sketch_files:
            raise ValueError(f"No validation sketches in {val_path}!")

    def _on_step(self) -> bool:
        # Trigger validation periodically
        if self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            print(f"\n--- Validation at step {self.num_timesteps} ---")
            results = []

            # Evaluate on each validation file
            for sketch_file in self.val_sketch_files:
                temp_config = {
                    **self.eval_env_config,
                    "specific_sketch_file": sketch_file,
                    "use_augmentation": False
                }
                # Create a temporary environment for evaluation
                eval_env = gym.make("DrawingEnv-v0", config=temp_config)
                obs, _ = eval_env.reset()
                done = False

                while not done:
                    # Predict action (non-deterministic for validation variety)
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, _, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated

                # Record results
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
                    "negative_reward": info.get("negative_reward"),
                    "jump_count": info.get("jump_count", 0),
                    "jump_draw_combo_count": info.get("jump_draw_combo_count", 0),
                    "episode_return": info.get("episode_return"),
                    "target_pixel_count": info.get("target_pixel_count", 0),
                })
                eval_env.close()

            self.validation_data.extend(results)
            self.logger.dump(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        """Saves validation results to CSV."""
        if self.validation_data:
            pd.DataFrame(self.validation_data).to_csv(self.save_path, index=False)
            print(f"[Validation] Saved data to {self.save_path}")


# ==========================================
# 2. TRAINING FUNCTION
# ==========================================
def run_training(config: dict):
    """
    Main function to set up directories, environment, model, and execute training.
    """
    # --- Configuration Extraction ---
    VERSION = config.get("VERSION", "_default_run")
    TOTAL_TIME_STEPS = config.get("TOTAL_TIME_STEPS", 5000000)
    LEARNING_RATE = config.get("LEARNING_RATE", 0.0003)
    ENT_COEF = config.get("ENT_COEF", 0.01)
    NUM_ENVS = config.get("NUM_ENVS", 1)
    BATCH_BASE_SIZE = config.get("BATCH_SIZE", 32)
    SEED = config.get("SEED", None)
    ENV_ID = config.get("ENV_ID", "DrawingEnv-v0")
    MODEL_NAME = config.get("MODEL_NAME", "drawing_agent_final.zip")

    env_config = config.get("ENV_CONFIG", {})
    cnn_padding = env_config.get("cnn_padding", True)
    validation_config = config.get("VALIDATION_CONFIG")

    print(f"Training on ENV_ID: {ENV_ID}")

    # --- Directory Setup ---
    BASE_OUTPUT_DIR = f"../training_outputs/{VERSION}/"
    LOG_DIR = os.path.join(BASE_OUTPUT_DIR, "logs/")
    MODELS_DIR = os.path.join(BASE_OUTPUT_DIR, "models/")
    TRAINING_DATA_PATH = os.path.join(BASE_OUTPUT_DIR, "training_data.csv")
    VALIDATION_DATA_PATH = os.path.join(BASE_OUTPUT_DIR, "validation_data.csv")
    STEP_DEBUG_DIR = os.path.join(BASE_OUTPUT_DIR, "step_debug/")

    # Configure environment debug path
    env_config["step_debug_path"] = STEP_DEBUG_DIR
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(STEP_DEBUG_DIR, exist_ok=True)

    # --- Environment Creation ---
    env = make_vec_env(
        ENV_ID,
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": env_config}
    )

    # --- Policy Setup ---
    policy_kwargs = dict(
        features_extractor_class=CustomCnnExtractor,
        features_extractor_kwargs=dict(features_dim=128, padding=cnn_padding),
    )

    # --- Model Loading / Initialization ---
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}. Loading...")
        model = PPO.load(model_path, env=env, tensorboard_log=LOG_DIR)
    else:
        print(f"Creating new model with CnnPolicy...")
        model = PPO(
            "CnnPolicy",
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

    # Store seed back to config for reference
    config["SEED"] = model.seed

    # --- Callback Registration ---
    callbacks = [TrainingDataCallback(save_path=TRAINING_DATA_PATH)]

    if validation_config:
        callbacks.append(ValidationCallback(
            eval_env_config=validation_config["ENV_CONFIG"],
            eval_freq=validation_config["EVAL_FREQ"],
            save_path=VALIDATION_DATA_PATH
        ))

    # --- Execution ---
    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=callbacks, reset_num_timesteps=False)

    # Save Final Model
    save_name = MODEL_NAME if not os.path.exists(model_path) else "new_" + MODEL_NAME
    model.save(os.path.join(MODELS_DIR, save_name))
    print(f"Model saved to {MODELS_DIR}")

    env.close()