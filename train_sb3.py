import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
# 【新增】导入 torch 库以设置其随机种子
import torch

# 导入您的环境和 stable-baselines3
from env import MultiAgentPathFindingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print("⚠️ 未能设置中文字体。图像中的中文可能无法正确显示。")
    print("   请确保您的系统已安装 'SimHei' 字体，或尝试使用 'Microsoft YaHei' 等其他中文字体。")


# ------------------- 单智能体包装器 (保持不变) -------------------
class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env: MultiAgentPathFindingEnv):
        super().__init__(env)
        self.env = env
        self.n_agents = self.env.n_agents
        original_obs_dim = self.env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_agents * original_obs_dim,),
            dtype=np.float32
        )
        original_action_dim = self.env.action_space.shape[0]
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(self.n_agents * original_action_dim,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        list_obs, info = self.env.reset(**kwargs)
        return np.concatenate(list_obs).flatten(), info

    def step(self, action):
        split_actions = np.split(action, self.n_agents)
        list_obs, list_rewards, list_dones, list_truncated, info = self.env.step(split_actions)
        obs = np.concatenate(list_obs).flatten()
        reward = sum(list_rewards)
        terminated = any(list_dones)
        truncated = any(list_truncated)
        return obs, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        return self.env.render(mode)

    def close(self):
        self.env.close()


# ------------------- 自定义回调函数 (保持不变) -------------------
class CustomCallback(BaseCallback):
    def __init__(self, eval_env,
                 eval_freq_episodes=100,
                 n_eval_episodes=1,
                 best_model_save_path='./sb3_best_model3/',
                 image_save_path='./evaluation_images3/',
                 verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq_episodes = eval_freq_episodes
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.image_save_path = image_save_path
        self.best_mean_reward = -np.inf
        self.episode_count = 0

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.image_save_path is not None:
            os.makedirs(self.image_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                if 'episode' in info:
                    self.episode_count += 1
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    print(f"Episode {self.episode_count} | Steps: {episode_length} | Reward: {episode_reward:.2f}")

                    if self.episode_count > 0 and self.episode_count % self.eval_freq_episodes == 0:
                        self._run_evaluation()
        return True

    def _run_evaluation(self):
        if self.verbose > 0:
            print(f"\n--- Running evaluation after episode {self.episode_count} (timestep {self.num_timesteps}) ---")

        # 【新增】为评估环境设置种子，确保评估过程也是可复现的
        # 注意：eval_env.seed() 是旧版gym的用法，新版gymnasium推荐在reset时设置
        # 但由于回调函数中无法直接修改，更稳妥的方式是在创建时就处理
        # 这里我们依赖于模型预测的确定性（deterministic=True）

        episode_rewards, episode_lengths = [], []
        for i in range(self.n_eval_episodes):
            # 可以在这里为每次评估重置种子，以确保每次评估都从完全相同的状态开始
            # obs, _ = self.eval_env.reset(seed=SEED + i)
            obs, _ = self.eval_env.reset()
            done = False
            total_reward, total_steps = 0, 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                total_steps += 1

            unwrapped_env = self.eval_env.unwrapped
            img = unwrapped_env.render(mode='rgb_array')
            if img is not None:
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                plt.axis('off')
                plt.title(
                    f'Eval Ep {i + 1} at Episode {self.episode_count}\nSteps: {total_steps} | Reward: {total_reward:.2f}')
                save_path = os.path.join(self.image_save_path, f'eval_ep{self.episode_count}_ep{i + 1}.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            episode_rewards.append(total_reward)
            episode_lengths.append(total_steps)

        mean_reward = np.mean(episode_rewards)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.dump(self.num_timesteps)

        if self.verbose > 0:
            print(f"Evaluation finished. Mean reward: {mean_reward:.2f}")

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print(f"🎉 New best model found! Saving to {self.best_model_save_path}")
            self.model.save(os.path.join(self.best_model_save_path, "best_model"))

        print("----------------------------------------------------")


def make_env():
    env = MultiAgentPathFindingEnv()
    env = SingleAgentWrapper(env)
    env = Monitor(env)
    return env


if __name__ == '__main__':
    # --- 【新增】设置随机种子以确保复现性 ---
    SEED = 26  # 您可以选择任何整数作为种子
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # -----------------------------------------

    # --- 1. 创建环境 ---
    # 【修改】为 make_vec_env 添加种子
    train_env = make_vec_env(make_env, n_envs=4, seed=SEED)

    # 评估环境的种子将在reset时被隐式地或显式地设置
    eval_env = make_env()

    # --- 2. 配置回调函数 (保持不变) ---
    callback = CustomCallback(
        eval_env,
        eval_freq_episodes=200,
        verbose=1
    )

    # --- 3. 创建并训练模型 ---
    # 【修改】为 SAC 模型添加种子
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log="./sb3_sac_tensorboard_logs/",
        # --- SAC 专用超参数 ---
        buffer_size=200_000,
        batch_size=512,
        learning_rate=1e-4,
        gamma=0.99,
        tau=0.005,
        learning_starts=10000,
        train_freq=(1, "step"),
        # 【新增】为模型设置种子
        seed=SEED
    )

    print(f"Starting training with Stable-Baselines3 SAC (Seed: {SEED})...")
    model.learn(total_timesteps=10_000_000, callback=callback)

    print("\nTraining finished.")