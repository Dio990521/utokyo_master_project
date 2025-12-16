import json
from main_train.train import run_training
import os
import numpy as np
from PIL import Image


def _load_sketch_from_path(filepath, canvas_size):
    sketch = Image.open(filepath).resize(canvas_size).convert('L')
    sketch_array = np.array(sketch)
    return (sketch_array / 255.0).astype(np.float32)

def preload_all_data(sketch_path, env_config):
    target_data_list = []
    empty_canvas = np.full(env_config["canvas_size"], 1.0, dtype=np.float32)
    canvas_size = env_config["canvas_size"]

    print(f"--- [Pre-loader] Starting pre-calculation for {sketch_path} ---")

    if not os.path.exists(sketch_path):
        raise ValueError(f"Sketch path not found: {sketch_path}")

    file_list = [f for f in os.listdir(sketch_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"--- [Pre-loader] Found {len(file_list)} images. Starting map calculations... ---")

    for i, filename in enumerate(file_list):
        filepath = os.path.join(sketch_path, filename)
        sketch_array = _load_sketch_from_path(filepath, canvas_size)
        target_data_list.append(sketch_array)

        if (i + 1) % 500 == 0 or i == len(file_list) - 1:
            print(f"  ...processed {i + 1}/{len(file_list)} images")

    print(f"--- [Pre-loader] Pre-calculation complete. Total data loaded: {len(target_data_list)} ---")
    return target_data_list

test1 = {
            "target_sketches_path": "../envs/drawing_env/training/8x8_sketches_train/",
            "val_sketches_path": "../envs/drawing_env/training/8x8_sketches_test/",
            "canvas_size": [8, 8],
            "max_steps": 64,
            "brush_size": 1,
            "use_combo": False,
            "combo_rate": 0.1,
            "penalty_scale_threshold": 1.4,
            "render_mode": None,
            "reward_correct": 1,
            "reward_wrong": -0.1,
            "repeat_scale": 0.5,
            "reward_jump": 0.1,
            "use_rook_move": False,
            "use_simplified_action_space": True,
            "use_dist_val_obs": True,
        }

test2 = {
            "target_sketches_path": "../envs/drawing_env/training/32x32_sketches_black_train/",
            "val_sketches_path": "../envs/drawing_env/training/32x32_sketches_black_test/",
            "canvas_size": [32, 32],
            "max_steps": 1024,
            "brush_size": 1,
            "use_combo": False,
            "combo_rate": 0.1,
            "penalty_scale_threshold": 0.4,
            "render_mode": None,
            "reward_correct": 1,
            "reward_wrong": -0.5,
            "repeat_scale": 0.5,
            "reward_jump": 0.5,
            "use_rook_move": False,
            "use_simplified_action_space": True,
            "use_dist_val_obs": True,
}

test3 = {
            "target_sketches_path": "../envs/drawing_env/training/32x32_sketches_black_train/",
            "val_sketches_path": "../envs/drawing_env/training/32x32_sketches_black_test/",
            "canvas_size": [32, 32],
            "max_steps": 1024,
            "brush_size": 1,
            "use_combo": False,
            "combo_rate": 0.1,
            "penalty_scale_threshold": 1.6,
            "render_mode": None,
            "reward_correct": 1,
            "reward_wrong": -0.5,
            "repeat_scale": 0.5,
            "reward_jump": 0.0,
            "use_rook_move": False,
            "use_simplified_action_space": True,
            "use_dist_val_obs": True,
}

experiments = [
    {
        "VERSION": "jump_test_8x8",
        "ENV_ID": "DrawingEnv-v0",
        "TOTAL_TIME_STEPS": 1000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 16,
        "BATCH_BASE_SIZE": 512,
        "ENT_COEF": 0.01,
        "ENV_CONFIG": test1,
        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 20000000,
            "ENV_CONFIG": test1,
        }
    },
    # {
    #     "VERSION": "20251217_black_threshold04_jump_near_reward05",
    #     "ENV_ID": "DrawingEnv-v0",
    #     "TOTAL_TIME_STEPS": 5000000,
    #     "LEARNING_RATE": 0.0003,
    #     "NUM_ENVS": 16,
    #     "BATCH_BASE_SIZE": 512,
    #     "ENT_COEF": 0.01,
    #     "ENV_CONFIG": test2,
    #     "VALIDATION_CONFIG": {
    #         "EVAL_FREQ": 20000000,
    #         "ENV_CONFIG": test2,
    #     }
    # },
    # {
    #     "VERSION": "20251217_black_threshold17_jump_near",
    #     "ENV_ID": "DrawingEnv-v0",
    #     "TOTAL_TIME_STEPS": 5000000,
    #     "LEARNING_RATE": 0.0003,
    #     "NUM_ENVS": 16,
    #     "BATCH_BASE_SIZE": 512,
    #     "ENT_COEF": 0.01,
    #     "ENV_CONFIG": test3,
    #     "VALIDATION_CONFIG": {
    #         "EVAL_FREQ": 20000000,
    #         "ENV_CONFIG": test3,
    #     }
    # },
]
if __name__ == '__main__':
    total_experiments = len(experiments)
    for i, config in enumerate(experiments):
        VERSION = config.get("VERSION", "_default_run")
        BASE_OUTPUT_DIR = f"../training_outputs/{VERSION}/"
        config_save_path = os.path.join(BASE_OUTPUT_DIR, "config.json")
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_save_path}")

    for i, config in enumerate(experiments):
        print(f"\n\n<<<<<<<<<< Starting Experiment {i + 1}/{total_experiments} >>>>>>>>>>")
        config["ENV_CONFIG"] = config["ENV_CONFIG"].copy()
        if config.get("VALIDATION_CONFIG"):
            config["VALIDATION_CONFIG"]["ENV_CONFIG"] = config["VALIDATION_CONFIG"]["ENV_CONFIG"].copy()
        try:
            run_training(config)
            print(f">>>>>>>>>> Experiment {config['VERSION']} Finished Successfully! <<<<<<<<<<")
        except Exception as e:
            print(f"!!!!!!!! Experiment {config['VERSION']} Failed with Error: {e} !!!!!!!!")
            import traceback
            traceback.print_exc()

    print("\n\nAll experiments have been completed!")