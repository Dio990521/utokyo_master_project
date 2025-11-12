import json

from main_train.train import run_training
import os
import numpy as np
from PIL import Image
from envs.drawing_env.tools.image_process import calculate_reward_map, calculate_dynamic_distance_map

TRAIN_SKETCH_DIR = "../envs/drawing_env/training/sketch_mix_augment/"
VALIDATION_SKETCH_DIR = "../envs/drawing_env/training/sketch_mix_augment/"

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

        reward_map = calculate_reward_map(
            sketch_array,
            env_config["reward_map_on_target"],
            env_config["reward_map_near_target"],
            env_config["reward_map_far_target"],
            env_config["reward_map_near_distance"]
        )

        initial_map = None
        if env_config["use_dynamic_distance_map_reward"]:
            initial_map = calculate_dynamic_distance_map(sketch_array, empty_canvas)

        # (sketch, reward_map, initial_map)
        target_data_list.append((sketch_array, reward_map, initial_map))

        if (i + 1) % 500 == 0 or i == len(file_list) - 1:
            print(f"  ...processed {i + 1}/{len(file_list)} images")

    print(f"--- [Pre-loader] Pre-calculation complete. Total data loaded: {len(target_data_list)} ---")
    return target_data_list

config_2squares_1 = {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "val_sketches_path": VALIDATION_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "brush_size": 1,
            "use_triangles": False,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
            "use_combo": True,
            "combo_rate": 0.2,
            "use_distance_map_obs": False,
            "use_dynamic_distance_map_reward": False,
            "navigation_reward_scale": 0.05,
            "reward_map_on_target": 0.1,
            "reward_map_near_target": -0.1,
            "reward_map_far_target": -0.1,
            "reward_map_near_distance": 2,
            "penalty_scale_threshold": 1.9,
            "use_budget_channel": False,
            "dynamic_budget_channel": False,
            "stroke_budget": 100,
            "use_stroke_reward": False,
            "r_stroke_hyper": 100,
            "stroke_reward_scale": 1.0,
            "render_mode": None,
            "similarity_weight": 0,
            "block_reward_scale": 0.0,
            "block_size": 8,
        }

config_mix_1 = {
            "target_sketches_path": "../envs/drawing_env/training/test/",
            "val_sketches_path": "../envs/drawing_env/training/test/",
            "canvas_size": [8, 8],
            "max_steps": 64,
            "use_time_penalty": False,
            "brush_size": 1,
            "use_triangles": False,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
            "use_combo": False,
            "combo_rate": 1.1,
            "use_distance_map_obs": False,
            "use_dynamic_distance_map_reward": False,
            "navigation_reward_scale": 0.05,
            "reward_map_on_target": 1.0,
            "reward_map_near_target": -1.0,
            "reward_map_far_target": -1.0,
            "reward_map_near_distance": 2,
            "penalty_scale_threshold": 1.9,
            "use_budget_channel": False,
            "dynamic_budget_channel": False,
            "stroke_budget": 100,
            "use_stroke_reward": False,
            "r_stroke_hyper": 100,
            "stroke_reward_scale": 1.0,
            "render_mode": None,
            "similarity_weight": 0,
            "block_reward_scale": 0.0,
            "block_size": 8,
        }

config_2squares_3 = {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "val_sketches_path": VALIDATION_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "brush_size": 3,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
            "use_combo": False,
            "use_dynamic_distance_map_reward": False,
            "navigation_reward_scale": 0.05,
            "reward_map_on_target": 0.1,
            "reward_map_near_target": -0.1,
            "reward_map_far_target": -0.1,
            "reward_map_near_distance": 2,
            "penalty_scale_threshold": 0.9,
            "use_budget_channel": True,
            "dynamic_budget_channel": False,
            "stroke_budget": 100,
            "use_stroke_reward": True,
            "r_stroke_hyper": 40,
            "stroke_reward_scale": 1.0,
            "render_mode": None,
            "similarity_weight": 0,
            "block_reward_scale": 0.0,
            "block_size": 8,
        }

config_2squares_4 = {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "val_sketches_path": VALIDATION_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "brush_size": 3,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
            "use_combo": False,
            "use_dynamic_distance_map_reward": False,
            "navigation_reward_scale": 0.05,
            "reward_map_on_target": 0.1,
            "reward_map_near_target": -0.1,
            "reward_map_far_target": -0.1,
            "reward_map_near_distance": 2,
            "penalty_scale_threshold": 0.9,
            "use_budget_channel": False,
            "dynamic_budget_channel": False,
            "stroke_budget": 100,
            "use_stroke_reward": True,
            "r_stroke_hyper": 100,
            "stroke_reward_scale": 1.0,
            "render_mode": None,
            "similarity_weight": 0,
            "block_reward_scale": 0.0,
            "block_size": 8,
        }

test1 = {
            "target_sketches_path": "../envs/drawing_env/training/sketch_num_augment/",
            "val_sketches_path": "../envs/drawing_env/training/sketch_num_augment/",
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "use_time_penalty": False,
            "brush_size": 3,
            "use_triangles": False,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
            "use_combo": False,
            "combo_rate": 1.1,
            "use_distance_map_obs": False,
            "use_dynamic_distance_map_reward": False,
            "navigation_reward_scale": 0.05,
            "reward_map_on_target": 0.1,
            "reward_map_near_target": -0.1,
            "reward_map_far_target": -0.1,
            "reward_map_near_distance": 2,
            "penalty_scale_threshold": 0.9,
            "use_budget_channel": False,
            "dynamic_budget_channel": False,
            "stroke_budget": 100,
            "use_stroke_reward": False,
            "r_stroke_hyper": 100,
            "stroke_reward_scale": 1.0,
            "render_mode": None,
            "similarity_weight": 0,
            "block_reward_scale": 0.0,
            "block_size": 8,
        }

test2 = {
            "target_sketches_path": "../envs/drawing_env/training/sketch_num_augment/",
            "val_sketches_path": "../envs/drawing_env/training/sketch_num_augment/",
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "use_time_penalty": False,
            "brush_size": 3,
            "use_triangles": False,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
            "use_combo": False,
            "combo_rate": 0.1,
            "use_distance_map_obs": False,
            "use_dynamic_distance_map_reward": False,
            "navigation_reward_scale": 0.05,
            "reward_map_on_target": 0.1,
            "reward_map_near_target": -0.1,
            "reward_map_far_target": -0.1,
            "reward_map_near_distance": 2,
            "penalty_scale_threshold": 0.8,
            "use_budget_channel": False,
            "dynamic_budget_channel": False,
            "stroke_budget": 100,
            "use_stroke_reward": False,
            "r_stroke_hyper": 100,
            "stroke_reward_scale": 1.0,
            "render_mode": None,
            "similarity_weight": 0,
            "block_reward_scale": 0.0,
            "block_size": 8,
        }

experiments = [
    # {
    #     "VERSION": "fps_test_9",
    #     "TOTAL_TIME_STEPS": 2000000,
    #     "LEARNING_RATE": 0.0003,
    #     "NUM_ENVS": 16,
    #     "BATCH_BASE_SIZE": 512,
    #     "ENT_COEF": 0.01,
    #     "ENV_CONFIG": test1,
    #     "VALIDATION_CONFIG": {
    #         "EVAL_FREQ": 2048 * 25,
    #         "ENV_CONFIG": test1,
    #     }
    # },
    # {
    #     "VERSION": "20251114_pen3x3_num_threshold09",
    #     "TOTAL_TIME_STEPS": 5000000,
    #     "LEARNING_RATE": 0.0003,
    #     "NUM_ENVS": 16,
    #     "BATCH_BASE_SIZE": 512,
    #     "ENT_COEF": 0.01,
    #     "ENV_CONFIG": test1,
    #     "VALIDATION_CONFIG": {
    #         "EVAL_FREQ": 5000000,
    #         "ENV_CONFIG": test1,
    #     }
    # },
    {
        "VERSION": "20251114_pen3x3_num_threshold08",
        "TOTAL_TIME_STEPS": 5000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 16,
        "BATCH_BASE_SIZE": 512,
        "ENT_COEF": 0.01,
        "ENV_CONFIG": test2,
        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 5000000,
            "ENV_CONFIG": test2,
        }
    },
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
        print("Pre-loading ALL training data...")
        PRECALCULATED_TRAIN_DATA = preload_all_data(config["ENV_CONFIG"]["target_sketches_path"], config["ENV_CONFIG"])
        print("\nPre-loading ALL validation data...")
        PRECALCULATED_VAL_DATA = preload_all_data(config["ENV_CONFIG"]["val_sketches_path"], config["ENV_CONFIG"])
        print(f"\n\n<<<<<<<<<< Starting Experiment {i+1}/{total_experiments} >>>>>>>>>>")
        config["ENV_CONFIG"]["precalculated_data"] = PRECALCULATED_TRAIN_DATA
        if config["VALIDATION_CONFIG"]:
            config["VALIDATION_CONFIG"]["ENV_CONFIG"]["precalculated_data"] = PRECALCULATED_VAL_DATA
        try:
            run_training(config)
            print(f">>>>>>>>>> Experiment {config['VERSION']} Finished Successfully! <<<<<<<<<<")
        except Exception as e:
            print(f"!!!!!!!! Experiment {config['VERSION']} Failed with Error: {e} !!!!!!!!")

    print("\n\nAll experiments have been completed!")