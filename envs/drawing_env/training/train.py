from stable_baselines3 import PPO # 或 SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from envs.drawing_env.draw_env import DrawingAgentEnv # 导入你的自定义环境
from envs.drawing_env.tools.image_process import ImageDraw
import os

LOG_DIR = "saved_logs/"
MODELS_DIR = "saved_models/"
SKETCH_DATA_PATH = "sketches/"
CANVAS_SIZE = (100, 100)
MAX_EPISODE_STEPS = 1000
TOTAL_TIME_STEPS = 1000000

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SKETCH_DATA_PATH, exist_ok=True)

env = make_vec_env(
    lambda: DrawingAgentEnv(
        target_sketches_path=SKETCH_DATA_PATH,
        config={
            "canvas_size": CANVAS_SIZE,
            "render": False,
            "max_steps": MAX_EPISODE_STEPS,
            "render_mode": "human",
            "mode": "training"
        }
    ),
    n_envs=1,
    seed=0
)

#env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    # features_extractor_class=CustomCNNFeaturesExtractor
)

model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=LOG_DIR,
    policy_kwargs=policy_kwargs,
)

eval_env = DrawingAgentEnv(
    target_sketches_path=SKETCH_DATA_PATH,
    config={
        "canvas_size": CANVAS_SIZE,
        "render": False,
        "max_steps": MAX_EPISODE_STEPS,
        "render_mode": None,
        "mode": "training"
    }
)
# eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99) # 评估环境也需要归一化

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODELS_DIR,
    log_path=LOG_DIR,
    eval_freq=MAX_EPISODE_STEPS * 10,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=MAX_EPISODE_STEPS * 100,
    save_path=MODELS_DIR,
    name_prefix="drawing_agent_checkpoint",
    save_replay_buffer=True,
    save_vecnormalize=True,
)


print("Start training...")
model.learn(
    total_timesteps=TOTAL_TIME_STEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True
)
print("Training finished.")

model.save(os.path.join(MODELS_DIR, "drawing_agent_final.zip"))
# if isinstance(env, VecNormalize):
#     env.save(os.path.join(MODELS_DIR, "vec_norm_final.pkl"))

print("Model saved to models/drawing_agent_final.zip")