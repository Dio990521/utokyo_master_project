from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.drawing_env.draw_env_grey import DrawingAgentGreyEnv
import os


VERSION = "20251203_pen3x3_width3_threshold04_combo01" #20251122_pen3x3trans1x1_width3_threshold04_redo_2
MODELS_DIR = f"../training_outputs/{VERSION}/models/"
SKETCH_DATA_PATH = "../envs/drawing_env/training/32x32_sketches_width3_test/"
CANVAS_SIZE = (32, 32)
MAX_EPISODE_STEPS = 1000

model_path = os.path.join(MODELS_DIR, "drawing_agent_final.zip")

def make_env():
    return DrawingAgentGreyEnv(
        config={
            "canvas_size": CANVAS_SIZE,
            "render": False,
            "max_steps": MAX_EPISODE_STEPS,
            "render_mode": "human",
            "target_sketches_path": SKETCH_DATA_PATH,
        }
    )

#eval_env = DummyVecEnv([make_env])
eval_env = DrawingAgentGreyEnv(
        config={
            "canvas_size": CANVAS_SIZE,
            "render": False,
            "max_steps": MAX_EPISODE_STEPS,
            "render_mode": "human",
            "target_sketches_path": SKETCH_DATA_PATH,
            "use_mvg_penalty_compensation": False,
            "brush_size": 3,
            "use_combo": False,
            "combo_rate": 1.1,
            "use_stroke_trajectory_obs": False,
            "use_distance_map_obs": False,
            "use_dynamic_distance_map_reward": False,
            "navigation_reward_scale": 0.05,
            "reward_map_on_target": 0.1,
            "reward_map_near_target": -0.1,
            "reward_map_far_target": -0.1,
            "reward_map_near_distance": 2,
            "penalty_scale_threshold": 0.8,
            "use_budget_channel": False,
            "dynamic_budget_channel": False,
            "stroke_budget": 0,
            "use_stroke_reward": False,
            "r_stroke_hyper": 0,
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
    action, _states = model.predict(obs, deterministic=False)
    eval_env.render()

    obs, reward, terminated, truncated, info = eval_env.step(action)
    print("action", action, "reward", reward)
    episode_reward += reward
    if terminated or truncated:
        break
print(info)

eval_env.close()