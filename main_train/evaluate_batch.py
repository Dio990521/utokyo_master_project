import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from envs.drawing_env.draw_env import DrawingAgentEnv
from envs.drawing_env.tools.custom_cnn import CustomCnnExtractor

VERSION = "final5_obs_tcr_action_j"
MODEL_PATH = f"../training_outputs/{VERSION}/models/drawing_agent_final.zip"
TEST_DATA_PATH = "../data/32x32_width1_test/"
OUTPUT_CSV = f"../training_outputs/{VERSION}/evaluation_results.csv"

CANVAS_SIZE = (32, 32)
MAX_EPISODE_STEPS = 1024
EVAL_TIMES = 20
USE_AUGMENTATION = False
BRUSH_SIZE = 1

ENV_CONFIG_TEMPLATE = {
    "canvas_size": CANVAS_SIZE,
    "render": False,
    "max_steps": MAX_EPISODE_STEPS,
    "render_mode": None,
    "brush_size": BRUSH_SIZE,
    "use_combo": False,
    "combo_rate": 1.1,
    "penalty_scale_threshold": 0.6,
    "reward_correct": 1,
    "reward_wrong": -0.25,
    "repeat_scale": 0,
    "reward_jump": 0,
    "jump_penalty": -0.25,
    "use_jump": True,
    "use_jump_penalty": True,
    "use_remaining_obs": True,
    "use_canvas_obs": True,
    "use_target_sketch_obs": True,
    "use_augmentation": USE_AUGMENTATION
}


def evaluate_all_images():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data path not found at {TEST_DATA_PATH}")
        return

    image_files = [f for f in os.listdir(TEST_DATA_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No image files found in test directory.")
        return

    print(f"Found {len(image_files)} images. Starting evaluation...")
    print(f"Eval Times per Image: {EVAL_TIMES}, Augmentation: {USE_AUGMENTATION}")

    results = []

    dummy_env = DrawingAgentEnv(config=ENV_CONFIG_TEMPLATE)
    model = PPO.load(MODEL_PATH, env=dummy_env)
    print(f"Model loaded successfully.")

    for img_file in tqdm(image_files, desc="Evaluating Images"):
        img_path = os.path.join(TEST_DATA_PATH, img_file)

        current_config = ENV_CONFIG_TEMPLATE.copy()
        current_config["specific_sketch_file"] = img_path

        eval_env = DrawingAgentEnv(config=current_config)

        for i in range(EVAL_TIMES):
            obs, _ = eval_env.reset()
            done = False
            truncated = False

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = eval_env.step(action)

            target_pixel_count = info.get("target_pixel_count", 1)
            jump_draw_combo_count = info.get("jump_draw_combo_count", 0)

            jump_ratio = 0.0
            if target_pixel_count > 0:
                jump_ratio = jump_draw_combo_count / target_pixel_count

            results.append({
                "image_name": img_file,
                "eval_idx": i + 1,
                "augmentation": USE_AUGMENTATION,
                "precision": info.get("precision", 0.0),
                "recall": info.get("recall_black", 0.0),
                "f1_score": info.get("f1_score", 0.0),
                "jump_ratio": jump_ratio,
                "jump_count": info.get("jump_count", 0),
                "jump_draw_combo_count": jump_draw_combo_count,
                "target_pixel_count": target_pixel_count,
                "total_painted": info.get("total_painted", 0),
                "correctly_painted": info.get("correctly_painted", 0),
                "episode_return": info.get("episode_return", 0)
            })

        eval_env.close()

    df = pd.DataFrame(results)

    print("\n=== Evaluation Summary ===")
    print(f"Total Episodes: {len(df)}")
    print(f"Average Precision: {df['precision'].mean():.4f}")
    print(f"Average Recall:    {df['recall'].mean():.4f}")
    print(f"Average F1 Score:  {df['f1_score'].mean():.4f}")
    print(f"Average Jump Ratio:{df['jump_ratio'].mean():.4f}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDetailed results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    evaluate_all_images()