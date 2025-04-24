from gymnasium.envs.registration import register

register(
    id='MouseClick-v0',
    entry_point='envs.mouse_click.click_env:ClickEnv',
    kwargs={"config": {"width": 640, "height": 480}}
)