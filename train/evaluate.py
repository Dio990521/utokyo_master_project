from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.drawing_env.draw_env import DrawingAgentEnv
import os


VERSION = "20251015_fix_budget10_1"
MODELS_DIR = f"../training_outputs/{VERSION}/models/"
SKETCH_DATA_PATH = "../envs/drawing_env/training/sketches/"
CANVAS_SIZE = (32, 32)
MAX_EPISODE_STEPS = 1000

model_path = os.path.join(MODELS_DIR, "drawing_agent_final.zip")

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
            "use_budget_channel": True,
            "dynamic_budget_channel": True,
            "terminate_on_budget_limit": True,
            "stroke_budget": 10,
            "use_local_reward_block": False,
            "local_reward_block_size": 3,
            "r_stroke_hyper": 100,
            "budget_weight": 1,
            "similarity_weight": 1,
            "mode": "training",
            "use_step_similarity_reward": False,
            "use_stroke_reward": False,
            "block_reward_scale": 0.0,
            "stroke_reward_scale": 1.0,
            "stroke_penalty": 0.0,
            "block_size": 8,
        }
    )

model = PPO.load(model_path, env=eval_env)
print(f"Model loaded from {model_path}")

obs, _ = eval_env.reset()
print(obs.shape)
eval_env.render()
episode_reward = 0
info = None
for step in range(MAX_EPISODE_STEPS):
    action, _states = model.predict(obs, deterministic=False)
    print("action", action)
    eval_env.render()

    obs, reward, terminated, truncated, info = eval_env.step(action)
    episode_reward += reward
    if terminated or truncated:
        break
print(info)

eval_env.close()