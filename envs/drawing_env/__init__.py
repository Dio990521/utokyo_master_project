from gymnasium.envs.registration import register

register(
    id="DrawingEnv-v0",
    entry_point="envs.drawing_env.draw_env:DrawingAgentEnv",
    kwargs={"config": {
        "canvas_size": [32, 32],
        "render": False,
        "rgb": True,
        "max_steps": 1000,
        "render_mode": None,
        "mode": "training"
        }
    }
)