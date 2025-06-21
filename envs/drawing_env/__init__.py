from gymnasium.envs.registration import register

register(
    id="DrawingEnv-v0",
    entry_point="custom_env.drawing_env:DrawingAgentEnv",
    max_episode_steps=1000,
    kwargs={"config": {
        "canvas_size": [100, 100],
        "render": False,
        "rgb": True,
        "render_mode": "human",
        "mode": "training"
        }
    }
)