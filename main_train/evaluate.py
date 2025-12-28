import numpy as np
from stable_baselines3 import PPO
from envs.drawing_env.draw_env import DrawingAgentEnv
from envs.drawing_env.draw_env_grey import DrawingAgentGreyEnv
import os
import matplotlib.pyplot as plt

SAVE_DIR = "../figures/drawing_process_figs_3"
VERSION = "final5_obs_r_action_j_09pt_continue"
MODELS_DIR = f"../training_outputs/{VERSION}/models/"
SKETCH_DATA_PATH = "../data/333/"
CANVAS_SIZE = (32, 32)
MAX_EPISODE_STEPS = 1024
ENV_ID = "DrawingEnv-v0" #DrawingEnv-v0, DrawingGreyEnv
save_drawing = True
#set initial position

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
            "penalty_scale_threshold": 0.9,
            "reward_correct": 1,
            "reward_wrong": -0.25,
            "repeat_scale": 0,
            "reward_jump": 0,
            "jump_penalty": -0.25,
            "jump_distance_threshold": 1.5,
            "use_jump": True,
            "use_jump_penalty": True,
            "use_remaining_obs": True,
            "use_canvas_obs": False,
            "use_target_sketch_obs": False,
            "use_augmentation": False
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
    canvas_img = eval_env.canvas  # shape: (H, W)

    if save_drawing and step % 10 == 0:
        plt.imsave(os.path.join(SAVE_DIR, f"step_{step:03d}.png"), canvas_img, cmap="gray")

    episode_reward += reward
    if terminated or truncated:
        if save_drawing:
            plt.imsave(os.path.join(SAVE_DIR, f"step_result.png"), canvas_img, cmap="gray")
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