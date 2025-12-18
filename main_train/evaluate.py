import numpy as np
from stable_baselines3 import PPO
from envs.drawing_env.draw_env import DrawingAgentEnv
from envs.drawing_env.draw_env_grey import DrawingAgentGreyEnv
import os
from PIL import Image


VERSION = "20251219_black_threshold04_jump_diff_obs_random_init" #20251210_black_threshold04_jump
MODELS_DIR = f"../training_outputs/{VERSION}/models/"
SKETCH_DATA_PATH = "../envs/drawing_env/training/32x32_sketches_width1_test/"
CANVAS_SIZE = (32, 32)
MAX_EPISODE_STEPS = 2048
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
            "reward_wrong": -0.2,
            "repeat_scale": 0,
            "reward_jump": 0,
            "jump_penalty": -0.2,
            "use_jump": True,
            "use_jump_penalty": True,
            "use_rook_move": False,
            "use_simplified_action_space": False,
            "use_dist_val_obs": False,
            "use_difference_obs": True,
            "use_canvas_obs": False,
            "use_target_sketch_obs": False
        }
    )



last_action = None
jump_draw_combo_count = 0
jump_count = 0

ACTION_JUMP = 18
ACTION_DRAW_IN_PLACE = 13

model = PPO.load(model_path, env=eval_env)
print("seed", model.seed)
print(f"Model loaded from {model_path}")

obs, _ = eval_env.reset()
eval_env.render()
episode_reward = 0
info = None
for step in range(MAX_EPISODE_STEPS):
    action, _states = model.predict(obs, deterministic=False)

    current_action = int(action)

    if last_action == ACTION_JUMP and current_action == ACTION_DRAW_IN_PLACE:
        jump_draw_combo_count += 1

    if current_action == ACTION_JUMP:
        jump_count += 1

    last_action = current_action

    eval_env.render()

    obs, reward, terminated, truncated, info = eval_env.step(action)
    print("action", action, "reward", reward)
    episode_reward += reward
    if terminated or truncated:
        break

print(info)
if hasattr(eval_env, 'target_sketch'):
    target_pixel_count = np.sum(eval_env.target_sketch < 0.5)
    print(f"\n[Statistics] Target Image Pixel Count (Ink): {target_pixel_count}")
    print(
        f"  > Total 'Jump -> Draw-in-Place' Combos: {jump_draw_combo_count}, {jump_draw_combo_count / target_pixel_count * 100:.2f}%")

else:
    print("\n[Statistics] Warning: Could not find target_sketch in env.")
eval_env.close()