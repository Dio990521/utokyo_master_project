from main_train.train import run_training
TRAIN_SKETCH_DIR = "../envs/drawing_env/training/sketch_num_augment/"
VALIDATION_SKETCH_DIR = "../envs/drawing_env/training/sketch_num_augment/"

config_2squares_1 = {
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

config_2squares_2 = {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "val_sketches_path": VALIDATION_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "brush_size": 1,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
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
            "brush_size": 1,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
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
            "brush_size": 1,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
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

test = {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "val_sketches_path": VALIDATION_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "brush_size": 1,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
            "use_dynamic_distance_map_reward": True,
            "navigation_reward_scale": 0.05,
            "reward_map_on_target": 0.5,
            "reward_map_near_target": -0.5,
            "reward_map_far_target": -0.5,
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

experiments = [
    {
        "VERSION": "20251105_aug_num_1",
        "TOTAL_TIME_STEPS": 5000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 16,
        "BATCH_BASE_SIZE": 512,
        "ENT_COEF": 0.01,
        "ENV_CONFIG": config_2squares_1,
        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 25,
            "ENV_CONFIG": config_2squares_1,
        }
    },
    {
        "VERSION": "20251105_aug_num_2",
        "TOTAL_TIME_STEPS": 5000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 16,
        "BATCH_BASE_SIZE": 512,
        "ENT_COEF": 0.01,
        "ENV_CONFIG": config_2squares_2,
        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 25,
            "ENV_CONFIG": config_2squares_2,
        }
    },
    {
        "VERSION": "20251105_aug_num_3",
        "TOTAL_TIME_STEPS": 5000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 16,
        "BATCH_BASE_SIZE": 512,
        "ENT_COEF": 0.01,
        "ENV_CONFIG": config_2squares_3,
        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 25,
            "ENV_CONFIG": config_2squares_3,
        }
    },
    {
        "VERSION": "20251105_aug_num_4",
        "TOTAL_TIME_STEPS": 5000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 16,
        "BATCH_BASE_SIZE": 512,
        "ENT_COEF": 0.01,
        "ENV_CONFIG": config_2squares_4,
        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 25,
            "ENV_CONFIG": config_2squares_4,
        }
    },
    # {
    #     "VERSION": "test",
    #     "TOTAL_TIME_STEPS": 5000000,
    #     "LEARNING_RATE": 0.0003,
    #     "NUM_ENVS": 16,
    #     "BATCH_BASE_SIZE": 512,
    #     "ENT_COEF": 0.01,
    #     "ENV_CONFIG": test,
    #     "VALIDATION_CONFIG": {
    #         "EVAL_FREQ": 2048 * 25,
    #         "ENV_CONFIG": test,
    #     }
    # },
]

if __name__ == '__main__':
    total_experiments = len(experiments)
    for i, config in enumerate(experiments):
        print(f"\n\n<<<<<<<<<< Starting Experiment {i+1}/{total_experiments} >>>>>>>>>>")
        try:
            run_training(config)
            print(f">>>>>>>>>> Experiment {config['VERSION']} Finished Successfully! <<<<<<<<<<")
        except Exception as e:
            print(f"!!!!!!!! Experiment {config['VERSION']} Failed with Error: {e} !!!!!!!!")

    print("\n\nAll experiments have been completed!")