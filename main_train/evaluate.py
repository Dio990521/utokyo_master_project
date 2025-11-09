from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.drawing_env.draw_env import DrawingAgentEnv
import os


VERSION = "20251110_pen3x3transfer1x1_num_combo_1" #20251107_pen3x3transfer1x1_num_1_redo1
MODELS_DIR = f"../training_outputs/{VERSION}/models/"
SKETCH_DATA_PATH = "../envs/drawing_env/training/sketch_num_augment/"
CANVAS_SIZE = (32, 32)
MAX_EPISODE_STEPS = 1000

model_path = os.path.join(MODELS_DIR, "drawing_agent_final_new.zip")

def make_env():
    return DrawingAgentEnv(
        config={
            "canvas_size": CANVAS_SIZE,
            "render": False,
            "max_steps": MAX_EPISODE_STEPS,
            "render_mode": "human",
            "target_sketches_path": SKETCH_DATA_PATH,
        }
    )

#eval_env = DummyVecEnv([make_env])
eval_env = DrawingAgentEnv(
        config={
            "canvas_size": CANVAS_SIZE,
            "render": False,
            "max_steps": MAX_EPISODE_STEPS,
            "render_mode": "human",
            "target_sketches_path": SKETCH_DATA_PATH,
            "brush_size": 1,
            "use_triangles": False,
            "num_rectangles": 2,
            "rect_min_width": 5,
            "rect_max_width": 15,
            "rect_min_height": 5,
            "rect_max_height": 15,
            "use_distance_map_obs": False,
            "use_combo": True,
            "use_dynamic_distance_map_reward": False,
            "navigation_reward_scale": 0.05,
            "reward_map_on_target": 0.5,
            "reward_map_near_target": -0.5,
            "reward_map_far_target": -0.5,
            "reward_map_near_distance": 2,
            "penalty_scale_threshold": 1.9,
            "use_budget_channel": False,
            "dynamic_budget_channel": False,
            "stroke_budget": 100,
            "use_stroke_reward": False,
            "r_stroke_hyper": 100,
            "stroke_reward_scale": 1.0,
            "similarity_weight": 0,
            "block_reward_scale": 0.0,
            "block_size": 8,
        }
    )

model = PPO.load(model_path, env=eval_env)
print("seed", model.seed)
print(f"Model loaded from {model_path}")

obs, _ = eval_env.reset()
print(obs.shape)
eval_env.render()
episode_reward = 0
info = None
for step in range(MAX_EPISODE_STEPS):
    action, _states = model.predict(obs, deterministic=True)
    eval_env.render()

    obs, reward, terminated, truncated, info = eval_env.step(action)
    print("action", action, "reward", reward)
    episode_reward += reward
    if terminated or truncated:
        break
print(info)

eval_env.close()