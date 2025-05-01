from gymnasium.envs.registration import register

register(
    id='MouseClick-v0',
    entry_point='envs.mouse_click.click_env:ClickEnv',
    kwargs={"config": {
        "width": 640,
        "height": 480,
        "render": False,
        "hp": 100,
        "reward_hit": 10.0,
        "total_targets": 100,
        "reward_miss": -0.1,
        "reward_success": 50.0,
        "reward_fail": -1.0}}
)