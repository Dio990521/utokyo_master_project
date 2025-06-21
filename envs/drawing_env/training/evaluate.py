from stable_baselines3 import PPO
from envs.drawing_env.draw_env import DrawingAgentEnv

import os

MODELS_DIR = "saved_models/"
SKETCH_DATA_PATH = "sketches/"
CANVAS_SIZE = (100, 100)
MAX_EPISODE_STEPS = 1000
RENDER_GIF_PATH = "results/drawing_process.gif"
NUM_EVAL_EPISODES = 1

os.makedirs("results", exist_ok=True)

model_path = os.path.join(MODELS_DIR, "best_model.zip")
# norm_env_path = os.path.join(MODELS_DIR, "vec_norm_final.pkl")

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}. Please train the agent first.")
    exit()

eval_env = DrawingAgentEnv(
        target_sketches_path=SKETCH_DATA_PATH,
        config={
            "canvas_size": CANVAS_SIZE,
            "render": False,
            "max_steps": MAX_EPISODE_STEPS,
            "render_mode": "human",
            "mode": "training"
        }
    )

# 如果训练时使用了 VecNormalize，评估时也需要加载
# if os.path.exists(norm_env_path):
#     eval_env = VecNormalize.load(norm_env_path, eval_env)
#     eval_env.training = False
#     eval_env.norm_reward = False

model = PPO.load(model_path, env=eval_env)
print(f"Model loaded from {model_path}")
obs, _ = eval_env.reset()
eval_env.render()
episode_reward = 0
for step in range(MAX_EPISODE_STEPS):
    action, _states = model.predict(obs, deterministic=False)

    eval_env.render()

    obs, reward, terminated, truncated, info = eval_env.step(action)
    episode_reward += reward

    if terminated or truncated:
        break

eval_env.close()