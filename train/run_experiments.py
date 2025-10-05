from train import run_training
TRAIN_SKETCH_DIR = "../envs/drawing_env/training/sketches/"
VALIDATION_SKETCH_DIR = "../envs/drawing_env/training/validation_sketches/"

experiments = [
    {
        "VERSION": "20251008_1",
        "TOTAL_TIME_STEPS": 10000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 5,
        "ENT_COEF": 0.01,

        "ENV_CONFIG": {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "stroke_budget": 100,
            "use_local_reward_block": False,
            "local_reward_block_size": 3,
            "r_stroke_hyper": 100,
            "render_mode": None,
            "budget_weight": 1,
            "similarity_weight": 1,
            "mode": "training",
            "use_step_similarity_reward": False,
            "use_stroke_reward": True,
            "block_reward_scale": 0.0,
            "stroke_reward_scale": 1.0,
            "stroke_penalty": 0.0,
            "block_size": 8,
        },

        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 50,
            "ENV_CONFIG": {
                "target_sketches_path": VALIDATION_SKETCH_DIR,
                "canvas_size": [32, 32],
                "max_steps": 1000,
                "stroke_budget": 100,
                "use_local_reward_block": False,
                "local_reward_block_size": 3,
                "r_stroke_hyper": 100,
                "render_mode": None,
                "budget_weight": 1,
                "similarity_weight": 1,
                "mode": "training",
                "use_step_similarity_reward": False,
                "use_stroke_reward": True,
                "block_reward_scale": 0.0,
                "stroke_reward_scale": 1.0,
                "stroke_penalty": 0.0,
                "block_size": 8,            }
        }
    },
    {
        "VERSION": "20251008_2",
        "TOTAL_TIME_STEPS": 10000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 5,
        "ENT_COEF": 0.01,

        "ENV_CONFIG": {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "stroke_budget": 100,
            "use_local_reward_block": False,
            "local_reward_block_size": 3,
            "r_stroke_hyper": 50,
            "render_mode": None,
            "budget_weight": 1,
            "similarity_weight": 1,
            "mode": "training",
            "use_step_similarity_reward": False,
            "use_stroke_reward": True,
            "block_reward_scale": 0.0,
            "stroke_reward_scale": 1.0,
            "stroke_penalty": 0.0,
            "block_size": 8,
        },

        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 50,
            "ENV_CONFIG": {
                "target_sketches_path": VALIDATION_SKETCH_DIR,
                "canvas_size": [32, 32],
                "max_steps": 1000,
                "stroke_budget": 50,
                "use_local_reward_block": False,
                "local_reward_block_size": 3,
                "r_stroke_hyper": 100,
                "render_mode": None,
                "budget_weight": 1,
                "similarity_weight": 1,
                "mode": "training",
                "use_step_similarity_reward": False,
                "use_stroke_reward": True,
                "block_reward_scale": 0.0,
                "stroke_reward_scale": 1.0,
                "stroke_penalty": 0.0,
                "block_size": 8, }
        }
    },
    {
        "VERSION": "20251008_3",
        "TOTAL_TIME_STEPS": 10000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 5,
        "ENT_COEF": 0.01,

        "ENV_CONFIG": {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "stroke_budget": 10,
            "use_local_reward_block": False,
            "local_reward_block_size": 3,
            "r_stroke_hyper": 100,
            "render_mode": None,
            "budget_weight": 1,
            "similarity_weight": 1,
            "mode": "training",
            "use_step_similarity_reward": False,
            "use_stroke_reward": True,
            "block_reward_scale": 0.0,
            "stroke_reward_scale": 1.0,
            "stroke_penalty": 0.0,
            "block_size": 8,
        },

        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 50,
            "ENV_CONFIG": {
                "target_sketches_path": VALIDATION_SKETCH_DIR,
                "canvas_size": [32, 32],
                "max_steps": 1000,
                "stroke_budget": 10,
                "use_local_reward_block": False,
                "local_reward_block_size": 3,
                "r_stroke_hyper": 100,
                "render_mode": None,
                "budget_weight": 1,
                "similarity_weight": 1,
                "mode": "training",
                "use_step_similarity_reward": False,
                "use_stroke_reward": True,
                "block_reward_scale": 0.0,
                "stroke_reward_scale": 1.0,
                "stroke_penalty": 0.0,
                "block_size": 8, }
        }
    },
    {
        "VERSION": "20251008_4",
        "TOTAL_TIME_STEPS": 10000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 5,
        "ENT_COEF": 0.01,

        "ENV_CONFIG": {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "stroke_budget": 100,
            "use_local_reward_block": False,
            "local_reward_block_size": 3,
            "r_stroke_hyper": 100,
            "render_mode": None,
            "budget_weight": 1,
            "similarity_weight": 1,
            "mode": "training",
            "use_step_similarity_reward": True,
            "use_stroke_reward": True,
            "block_reward_scale": 0.0,
            "stroke_reward_scale": 1.0,
            "stroke_penalty": 0.0,
            "block_size": 8,
        },

        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 50,
            "ENV_CONFIG": {
                "target_sketches_path": VALIDATION_SKETCH_DIR,
                "canvas_size": [32, 32],
                "max_steps": 1000,
                "stroke_budget": 100,
                "use_local_reward_block": False,
                "local_reward_block_size": 3,
                "r_stroke_hyper": 100,
                "render_mode": None,
                "budget_weight": 1,
                "similarity_weight": 1,
                "mode": "training",
                "use_step_similarity_reward": True,
                "use_stroke_reward": True,
                "block_reward_scale": 0.0,
                "stroke_reward_scale": 1.0,
                "stroke_penalty": 0.0,
                "block_size": 8, }
        }
    },
    {
        "VERSION": "20251008_5",
        "TOTAL_TIME_STEPS": 10000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 5,
        "ENT_COEF": 0.01,

        "ENV_CONFIG": {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "stroke_budget": 100,
            "use_local_reward_block": False,
            "local_reward_block_size": 3,
            "r_stroke_hyper": 50,
            "render_mode": None,
            "budget_weight": 1,
            "similarity_weight": 1,
            "mode": "training",
            "use_step_similarity_reward": True,
            "use_stroke_reward": True,
            "block_reward_scale": 0.0,
            "stroke_reward_scale": 1.0,
            "stroke_penalty": 0.0,
            "block_size": 8,
        },

        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 50,
            "ENV_CONFIG": {
                "target_sketches_path": VALIDATION_SKETCH_DIR,
                "canvas_size": [32, 32],
                "max_steps": 1000,
                "stroke_budget": 50,
                "use_local_reward_block": False,
                "local_reward_block_size": 3,
                "r_stroke_hyper": 100,
                "render_mode": None,
                "budget_weight": 1,
                "similarity_weight": 1,
                "mode": "training",
                "use_step_similarity_reward": True,
                "use_stroke_reward": True,
                "block_reward_scale": 0.0,
                "stroke_reward_scale": 1.0,
                "stroke_penalty": 0.0,
                "block_size": 8, }
        }
    },
    {
        "VERSION": "20251008_6",
        "TOTAL_TIME_STEPS": 10000000,
        "LEARNING_RATE": 0.0003,
        "NUM_ENVS": 5,
        "ENT_COEF": 0.01,

        "ENV_CONFIG": {
            "target_sketches_path": TRAIN_SKETCH_DIR,
            "canvas_size": [32, 32],
            "max_steps": 1000,
            "stroke_budget": 10,
            "use_local_reward_block": False,
            "local_reward_block_size": 3,
            "r_stroke_hyper": 100,
            "render_mode": None,
            "budget_weight": 1,
            "similarity_weight": 1,
            "mode": "training",
            "use_step_similarity_reward": True,
            "use_stroke_reward": True,
            "block_reward_scale": 0.0,
            "stroke_reward_scale": 1.0,
            "stroke_penalty": 0.0,
            "block_size": 8,
        },

        "VALIDATION_CONFIG": {
            "EVAL_FREQ": 2048 * 50,
            "ENV_CONFIG": {
                "target_sketches_path": VALIDATION_SKETCH_DIR,
                "canvas_size": [32, 32],
                "max_steps": 1000,
                "stroke_budget": 10,
                "use_local_reward_block": False,
                "local_reward_block_size": 3,
                "r_stroke_hyper": 100,
                "render_mode": None,
                "budget_weight": 1,
                "similarity_weight": 1,
                "mode": "training",
                "use_step_similarity_reward": True,
                "use_stroke_reward": True,
                "block_reward_scale": 0.0,
                "stroke_reward_scale": 1.0,
                "stroke_penalty": 0.0,
                "block_size": 8, }
        }
    },
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