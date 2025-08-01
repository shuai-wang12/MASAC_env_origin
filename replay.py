import time
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # 导入绘图库

from env import MultiAgentPathFindingEnv
from stable_baselines3 import SAC
from train_sb3 import SingleAgentWrapper

# --- 配置 ---
MODEL_PATH = "sb3_best_model3/best_model.zip"
# 【修改】我们将只回放一个回合，以生成清晰的曲线图
N_EPISODES = 1


def plot_results(timestamps, positions, velocities, n_agents):
    """
    绘制时空曲线和速度曲线
    :param timestamps: 时间戳列表
    :param positions: 包含每个智能体位置列表的列表
    :param velocities: 包含每个智能体速度列表的列表
    :param n_agents: 智能体数量
    """
    agent_colors = ['blue', 'red', 'green']

    # 解决中文显示问题
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("未能设置中文字体，图像中的中文可能无法正确显示。")

    # --- 1. 绘制时空曲线 (位置 vs. 时间) ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle('时空曲线 (位置 vs. 时间)', fontsize=16)

    for i in range(n_agents):
        x_coords = [pos[0] for pos in positions[i]]
        y_coords = [pos[1] for pos in positions[i]]
        ax1.plot(timestamps, x_coords, label=f'智能体 {i + 1} X坐标', color=agent_colors[i])
        ax2.plot(timestamps, y_coords, label=f'智能体 {i + 1} Y坐标', color=agent_colors[i], linestyle='--')

    ax1.set_ylabel('X 坐标')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('Y 坐标')
    ax2.legend()
    ax2.grid(True)

    plt.savefig("spacetime_curves.png")
    print("时空曲线图已保存为: spacetime_curves.png")
    plt.close(fig1)

    # --- 2. 绘制速度曲线 (速度 vs. 时间) ---
    plt.figure(figsize=(12, 6))
    plt.title('速度曲线 (速率 vs. 时间)', fontsize=16)
    for i in range(n_agents):
        plt.plot(timestamps, velocities[i], label=f'智能体 {i + 1} 速率', color=agent_colors[i])

    plt.xlabel('时间 (s)')
    plt.ylabel('速率 (units/s)')
    plt.legend()
    plt.grid(True)

    plt.savefig("velocity_curves.png")
    print("速度曲线图已保存为: velocity_curves.png")
    plt.close()


def evaluate_policy():
    """
    加载并回放策略，同时收集数据并绘图。
    """
    print(f"正在从 '{MODEL_PATH}' 加载模型...")

    env = MultiAgentPathFindingEnv()
    env = SingleAgentWrapper(env)
    model = SAC.load(MODEL_PATH, env=env)

    # 从环境中获取智能体数量和时间间隔
    n_agents = env.unwrapped.n_agents
    dt = env.unwrapped.interval

    for episode in range(N_EPISODES):
        print(f"\n--- 开始回放第 {episode + 1}/{N_EPISODES} 回合 (并记录数据) ---")
        obs, info = env.reset()

        # --- 【新增】初始化用于存储数据的列表 ---
        last_positions = np.array(env.unwrapped.agent_positions)
        positions_over_time = [[pos] for pos in last_positions]
        velocities_over_time = [[0.0] for _ in range(n_agents)]  # 初始速度为0
        timestamps = [0.0]

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render(mode='human')

            # --- 【新增】数据收集和计算 ---
            current_positions = np.array(env.unwrapped.agent_positions)
            # 计算速度向量: (当前位置 - 上一位置) / 时间间隔
            velocity_vectors = (current_positions - last_positions) / dt
            # 计算速率 (速度向量的大小)
            speeds = np.linalg.norm(velocity_vectors, axis=1)

            # 记录数据
            timestamps.append(env.unwrapped.current_step * dt)
            for i in range(n_agents):
                positions_over_time[i].append(current_positions[i])
                velocities_over_time[i].append(speeds[i])

            # 更新上一位置
            last_positions = current_positions

            time.sleep(0.05)

            if terminated or truncated:
                print("回合结束。正在生成曲线图...")
                # 【新增】调用绘图函数
                plot_results(timestamps, positions_over_time, velocities_over_time, n_agents)
                time.sleep(2)
                break

    env.close()
    print("\n所有回放和绘图已完成。")


if __name__ == '__main__':
    evaluate_policy()