from train import run_training

experiments = [
    {
        "VERSION": "exp_lr_high_0.001",
        "TOTAL_TIME_STEPS": 1000000,
        "LEARNING_RATE": 0.001,
    },
    {
        "VERSION": "exp_lr_medium_0.0003",
        "TOTAL_TIME_STEPS": 1000000,
        "LEARNING_RATE": 0.0003,
    },
    {
        "VERSION": "exp_lr_low_0.00005",
        "TOTAL_TIME_STEPS": 1000000,
        "LEARNING_RATE": 0.00005,
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