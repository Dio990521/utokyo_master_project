from train import run_training

experiments = [
    {
        "VERSION": "_1",
        "TOTAL_TIME_STEPS": 5000000,
        "LEARNING_RATE": 0.0003,
        "ENT_COEF": 0.02,
        "ENV_CONFIG": {
            "canvas_size": [32, 32],
            "render": False,
            "max_steps": 1000,
            "stroke_budget": 1,
            "render_mode": None,
            "budget_weight": 1,
            "similarity_weight": 1,
            "mode": "training",
            "use_step_similarity_reward": True,
            "target_sketches_path": "../envs/drawing_env/training/sketches/",
        }
    },
    {
        "VERSION": "_2",
        "TOTAL_TIME_STEPS": 5000000,
        "LEARNING_RATE": 0.0003,
        "ENT_COEF": 0.02,
        "ENV_CONFIG": {
            "canvas_size": [32, 32],
            "render": False,
            "max_steps": 1000,
            "stroke_budget": 1,
            "render_mode": None,
            "budget_weight": 1,
            "similarity_weight": 1,
            "mode": "training",
            "use_step_similarity_reward": False,
            "target_sketches_path": "../envs/drawing_env/training/sketches/",
        }
    },
]

total_experiments = len(experiments)
for i, config in enumerate(experiments):
    print(f"\n\n<<<<<<<<<< Starting Experiment {i+1}/{total_experiments} >>>>>>>>>>")
    try:
        run_training(config)
        print(f">>>>>>>>>> Experiment {config['VERSION']} Finished Successfully! <<<<<<<<<<")
    except Exception as e:
        print(f"!!!!!!!! Experiment {config['VERSION']} Failed with Error: {e} !!!!!!!!")

print("\n\nAll experiments have been completed!")