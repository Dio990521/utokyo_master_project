from stable_baselines3 import PPO
from envs.drawing_env.draw_env import DrawingAgentEnv
from envs.drawing_env.draw_env_grey import DrawingAgentGreyEnv
import os


VERSION = "20251210_black_threshold04_jump"
MODELS_DIR = f"../training_outputs/{VERSION}/models/"
SKETCH_DATA_PATH = "../envs/drawing_env/training/32x32_sketches_black_mix_test/"
CANVAS_SIZE = (32, 32)
MAX_EPISODE_STEPS = 1024
ENV_ID = "DrawingEnv-v0" #DrawingEnv-v0, DrawingGreyEnv

model_path = os.path.join(MODELS_DIR, "drawing_agent_final.zip")
if ENV_ID == "DrawingGreyEnv-v0":
    eval_env = DrawingAgentGreyEnv(
        config={
            "canvas_size": CANVAS_SIZE,
            "max_steps": MAX_EPISODE_STEPS,
            "render_mode": "human",
            "target_sketches_path": SKETCH_DATA_PATH,
            "brush_size": 1,
            "use_combo": False,
            "combo_rate": 1.1,
            "penalty_scale_threshold": 0.6,
            "use_difference_map_obs": False,
            "reward_correct": 1,
            "reward_wrong": -0.5,
            "use_multi_discrete": False,
            "use_coord_conv": False,
            "use_distance_reward": True,
            "distance_reward_scale": 0.1,
            "use_jump": True
        }
    )
else:
    eval_env = DrawingAgentEnv(
        config={
            "canvas_size": CANVAS_SIZE,
            "render": False,
            "max_steps": MAX_EPISODE_STEPS,
            "render_mode": "human",
            "target_sketches_path": SKETCH_DATA_PATH,
            "brush_size": 1,
            "use_combo": False,
            "combo_rate": 1.1,
            "penalty_scale_threshold": 0.4,
            "reward_correct": 1,
            "reward_wrong": -0.5,
            "repeat_scale": 0.5,
            "reward_jump": 0.1,
            "use_jump": True,
            "use_rook_move": False,
            "use_simplified_action_space": False,
            "use_dist_val_obs": False,
        }
    )
model = PPO.load(model_path, env=eval_env)
print("seed", model.seed)
print(f"Model loaded from {model_path}")

obs, _ = eval_env.reset()
eval_env.render()
episode_reward = 0
info = None
for step in range(MAX_EPISODE_STEPS):
    action, _states = model.predict(obs, deterministic=False)
    eval_env.render()

    obs, reward, terminated, truncated, info = eval_env.step(action)
    print("action", eval_env._decode_action(action), "reward", reward)
    episode_reward += reward
    if terminated or truncated:
        break
print(info)

eval_env.close()