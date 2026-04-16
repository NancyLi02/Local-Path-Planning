from stable_baselines3.common.env_checker import check_env
from light_weight_simulator import LocalPlannerEnv

env = LocalPlannerEnv()
check_env(env, warn=True, skip_render_check=True)

obs, info = env.reset(seed=0)
print("obs shape:", obs.shape)
print("action space:", env.action_space)
print("observation space:", env.observation_space)

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(reward, terminated, truncated, info)
    if terminated or truncated:
        obs, info = env.reset()