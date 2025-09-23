from train import run_training

experiments = [
    {
        "VERSION": "_20251001_1",
        "TOTAL_TIME_STEPS": 5000000,
        "LEARNING_RATE": 0.0003,
        "ENT_COEF": 0.01,
        "ENV_CONFIG": {
            "canvas_size": [32, 32],
            "render": False,
            "max_steps": 1000,
            "stroke_budget": 100,
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
            "target_sketches_path": "../envs/drawing_env/training/sketches/",
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