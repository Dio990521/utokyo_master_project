import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.drawing_env.draw_env import DrawingAgentEnv

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
VERSION = "final5_obs_r_action_j_09pt"
ENV_ID = "DrawingEnv-v0"

# File Paths
MODELS_DIR = f"../training_outputs/{VERSION}/models/"
SKETCH_DATA_PATH = "../data/333/"
SAVE_DIR = "../figures/drawing_process_figs_3"
MODEL_PATH = os.path.join(MODELS_DIR, "drawing_agent_final.zip")

# Environment Settings
CANVAS_SIZE = (32, 32)
MAX_EPISODE_STEPS = 1024
SAVE_DRAWING = False  # Set to True to save images of the drawing process

# Action Definitions (for statistics)
ACTION_JUMP = 18
ACTION_DRAW_IN_PLACE = 13


# ==========================================
# 2. MAIN EXECUTION
# ==========================================
def main():
    # --- Environment Initialization ---
    env_config = {
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

    eval_env = DrawingAgentEnv(config=env_config)

    # --- Load Model ---
    print(f"Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, env=eval_env)
    print("Model Seed:", model.seed)

    # --- Simulation Setup ---
    obs, _ = eval_env.reset()
    eval_env.render()

    episode_reward = 0
    info = None

    # Tracking variables
    last_action = None
    jump_draw_combo_count = 0
    jump_count = 0

    # Ensure output directory exists if saving images
    if SAVE_DRAWING and not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Main Loop ---
    for step in range(MAX_EPISODE_STEPS):
        # Predict action (deterministic=False for exploration/variety)
        action, _states = model.predict(obs, deterministic=False)
        current_action = int(action)

        # Track "Jump -> Draw" combos
        if last_action == ACTION_JUMP and current_action == ACTION_DRAW_IN_PLACE:
            jump_draw_combo_count += 1

        if current_action == ACTION_JUMP:
            jump_count += 1

        last_action = current_action

        # Update Environment
        eval_env.render()
        obs, reward, terminated, truncated, info = eval_env.step(action)
        print(f"Step: {step}, Action: {action}, Reward: {reward:.4f}")

        # Save intermediate canvas states
        canvas_img = eval_env.canvas
        if SAVE_DRAWING and step % 10 == 0:
            plt.imsave(os.path.join(SAVE_DIR, f"step_{step:03d}.png"), canvas_img, cmap="gray")

        episode_reward += reward

        # Check for episode end
        if terminated or truncated:
            if SAVE_DRAWING:
                plt.imsave(os.path.join(SAVE_DIR, "step_result.png"), canvas_img, cmap="gray")
            break

    # --- Final Statistics ---
    print(f"\nFinal Info: {info}")

    if hasattr(eval_env, 'target_sketch'):
        # Count black pixels (< 0.5) in the target image
        target_pixel_count = np.sum(eval_env.target_sketch < 0.5)

        # Avoid division by zero
        ratio = (jump_draw_combo_count / target_pixel_count * 100) if target_pixel_count > 0 else 0

        print(f"\n[Statistics] Target Image Pixel Count (Ink): {target_pixel_count}")
        print(f"  > Total 'Jump -> Draw-in-Place' Combos: {jump_draw_combo_count} ({ratio:.2f}%)")
    else:
        print("\n[Statistics] Warning: Could not find target_sketch in env.")

    eval_env.close()


if __name__ == "__main__":
    main()