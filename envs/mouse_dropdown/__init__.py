from gymnasium.envs.registration import register

register(
    id='MouseDropdown-v0',
    entry_point='envs.mouse_dropdown.drag_env:DropDownMenuEnv',
    kwargs={"config": {
        "width": 640,
        "height": 480,
        "action_space_mode": "simple",
        "render": False,
        "rgb": True,
        "obs_compress": False,
        "mode": "training",
        "max_hp": 10000000,
        "total_targets": 1,
        "max_step": 2000,
        "obs_mode": "simple"}
    }
)