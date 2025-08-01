import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 导入A*路径规划器
from astar import AstarPathFinder, AstarFeatureExtractor


def astar(start, goal, obstacles, grid_size):
    """A*路径规划的包装函数"""
    pathfinder = AstarPathFinder(grid_size)
    return pathfinder.get_path(start, goal, obstacles)


def sample_waypoints(path, n_points):
    """从路径中采样路标点，如果路径点不足，则用最后一个点填充"""
    if path is None or (isinstance(path, (list, np.ndarray)) and len(path) == 0):  # 正确处理空路径
        return [np.zeros(2) for _ in range(n_points)]

    path = np.array(path)  # 确保path是numpy数组
    if len(path) <= n_points:
        # 如果路径点不足，用最后一个点填充
        sampled = list(path)
        last_point = path[-1].copy()
        while len(sampled) < n_points:
            sampled.append(last_point.copy())
        return sampled

    indices = np.linspace(0, len(path) - 1, n_points, dtype=int)
    return [path[i].copy() for i in indices]


def get_next_waypoint(current_pos, path, threshold=0.5):
    """获取下一个路标点"""
    if not path:
        return None

    # 找到当前位置最近的路径点
    min_dist = float('inf')
    nearest_idx = 0
    for i, point in enumerate(path):
        dist = np.linalg.norm(current_pos - point)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i

    # 返回路径上的下一个点
    next_idx = min(nearest_idx + 1, len(path) - 1)
    return path[next_idx]


def dist_point_to_segment(p, a, b):
    """计算点到线段的距离"""
    # 向量化计算
    ap = p - a
    ab = b - a

    # 计算投影
    proj_len = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-6)
    proj_len = np.clip(proj_len, 0, 1)

    # 计算最近点
    closest = a + proj_len * ab

    return np.linalg.norm(p - closest)


class MultiAgentPathFindingEnv(gym.Env):
    """
    多智能体路径规划环境
    分阶段控制：
    1. 巷道内：使用固定规则移动到出口
    2. 巷道外：使用强化学习到达目标
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        super().__init__()

        # 环境参数
        self.grid_size = 10  # 网格大小
        self.n_agents = 3  # 智能体数量
        self.max_steps = 400  # 最大步数
        self.current_step = 0  # 当前步数
        self.interval = 0.1  # 时间间隔
        self.agent_radius = 0.3  # 智能体半径
        self.n_waypoints = 5  # A*路径上的路标点数量

        # 巷道出口位置（门口）
        self.exit_positions = [
            np.array([1.6, 6.6]),  # 每个智能体的出口位置
            np.array([1.6, 6.6]),
            np.array([1.6, 6.6])
        ]

        # 阶段控制
        self.agent_phases = ['tunnel'] * self.n_agents  # 'tunnel' 或 'rl'
        self.tunnel_speed = 0.5  # 巷道内的固定速度

        # 定义观察空间
        # 阶段二（RL阶段）的观察包括：
        # 1. 自身位置(x,y)：2维
        # 2. 目标位置(x,y)：2维
        # 3. 其他智能体信息:
        #    - 当前位置(x,y)
        #    - 相对速度(x,y)
        #    每个其他智能体4维，共(n_agents-1)*4维
        # 4. A*路径信息：
        #    - n_waypoints个路标点的相对位置(x,y)和距离(1)
        #    每个路标点3维，共n_waypoints*3维

        # 计算观察空间维度
        self_pos_dim = 2  # 自身位置
        goal_pos_dim = 2  # 目标位置
        other_agents_dim = (self.n_agents - 1) * 4  # 其他智能体信息
        waypoints_dim = self.n_waypoints * 3  # A*路径信息

        obs_dim = self_pos_dim + goal_pos_dim + other_agents_dim + waypoints_dim

        self.observation_space = spaces.Box(
            low=np.array([-self.grid_size] * obs_dim),
            high=np.array([self.grid_size] * obs_dim),
            dtype=np.float32
        )

        # 状态空间与观察空间相同
        self.state_space = self.observation_space

        # 连续动作空间：前进和转向
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # 为MADDPG设置必要的环境信息
        self.env_name = "MultiAgentPathFinding"
        self.state_dim = obs_dim
        self.action_dim = 2
        self.if_discrete = False

        # 初始化起点和终点位置
        self.start_positions = [
            np.array([1.75, 7]),  # 起始位置
            np.array([2.08, 8]),
            np.array([2.41, 9])
        ]

        self.goal_positions = [
            np.array([4, 1]),
            np.array([7, 1]),
            np.array([9, 4])
        ]

        # 初始化智能体状态
        self.agent_positions = None
        self.trajectories = [[] for _ in range(self.n_agents)]
        self.agent_directions = [0.0] * self.n_agents
        self.agent_reached_goal_flag = [False] * self.n_agents
        self.astar_paths = [[] for _ in range(self.n_agents)]
        self.previous_positions = None
        # 【新增】用于存储上一时刻的动作，以计算速度变化
        self.previous_actions = None

        # 定义线段障碍物
        self.obstacle_lines = [
            [np.array([2.0, 10.0]), np.array([0.0, 4.0])],
            [np.array([0.0, 4.0]), np.array([0.0, 3.0])],
            [np.array([0.0, 3.0]), np.array([1.0, 1.0])],
            [np.array([1.0, 1.0]), np.array([3.0, 0.0])],
            [np.array([3.0, 0.0]), np.array([7.0, 0.0])],
            [np.array([7.0, 0.0]), np.array([10.0, 1.0])],
            [np.array([10.0, 1.0]), np.array([10.0, 5.0])],
            [np.array([10.0, 5.0]), np.array([6.0, 6.0])],
            [np.array([6.0, 6.0]), np.array([4.0, 7.0])],
            [np.array([4.0, 7.0]), np.array([5.0, 10.0])],
            [np.array([5.0, 10.0]), np.array([2.0, 10.0])],
            [np.array([3.5, 10.0]), np.array([2.5, 7.0])]
        ]

        # 为 'human' 模式渲染准备的图形对象
        self.fig_human = None
        self.ax_human = None

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 重置基本状态
        self.current_step = 0
        self.agent_positions = [pos.copy() for pos in self.start_positions]
        self.agent_reached_goal_flag = [False] * self.n_agents
        self.previous_positions = [pos.copy() for pos in self.agent_positions]
        # 【修改】重置上一时刻的动作为零
        self.previous_actions = [np.zeros(self.action_space.shape[0]) for _ in range(self.n_agents)]
        self.agent_phases = ['tunnel'] * self.n_agents

        self.agent_directions = []
        # 定义一个所有智能体都朝向的目标点
        direction_target_point = np.array([1.62, 6.0])
        for i in range(self.n_agents):
            # 计算从智能体起始位置到目标点的向量
            direction_vector = direction_target_point - self.start_positions[i]
            # 使用arctan2(y, x)计算角度
            angle = np.arctan2(direction_vector[1], direction_vector[0])
            self.agent_directions.append(angle)

        # 清空轨迹
        self.trajectories = [[] for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            self.trajectories[i].append(self.agent_positions[i].copy())

        # 初始化A*路径为空（巷道阶段不需要A*路径）
        self.astar_paths = [None] * self.n_agents

        # 获取初始观察
        observations = self._get_observations()
        info = {}
        return observations, info

    def _check_collision(self, pos, agent_idx):
        """检查碰撞
        返回:
            collision_type: 0-无碰撞, 1-与其他智能体碰撞, 2-与墙壁碰撞
        """
        # 与其他智能体碰撞
        for i in range(self.n_agents):
            if i != agent_idx:
                if np.linalg.norm(pos - self.agent_positions[i]) < 2 * self.agent_radius:
                    return 1

        # 与障碍物碰撞
        for line_start, line_end in self.obstacle_lines:
            if dist_point_to_segment(pos, line_start, line_end) < self.agent_radius:
                return 2

        return 0

    def step(self, actions):
        """环境步进"""
        self.current_step += 1

        # 更新智能体位置
        new_positions = []
        rewards = []
        collision_occurred = False  # 碰撞标志
        collision_type = 0  # 碰撞类型

        for i in range(self.n_agents):
            if self.agent_reached_goal_flag[i]:
                # 已到达目标，保持位置不变
                new_positions.append(self.goal_positions[i].copy())
                rewards.append(0.0)
                continue

            if self.agent_phases[i] == 'tunnel':
                # 'tunnel'阶段完全忽略外部传入的action，只执行内部规则
                direction = np.arctan2(self.exit_positions[i][1] - self.agent_positions[i][1],
                                       self.exit_positions[i][0] - self.agent_positions[i][0])
                new_pos = self.agent_positions[i] + self.tunnel_speed * np.array([
                    np.cos(direction),
                    np.sin(direction)
                ]) * self.interval

                # 检查是否到达出口
                if np.linalg.norm(new_pos - self.exit_positions[i]) < 0.1:
                    self.agent_phases[i] = 'rl'  # 切换到RL阶段
                    new_pos = self.exit_positions[i].copy()
                    # 计算从出口到目标的A*路径
                    path = astar(
                        start=new_pos,
                        goal=self.goal_positions[i],
                        obstacles=self.obstacle_lines,
                        grid_size=self.grid_size
                    )

                    self.astar_paths[i] = path

                # 检查碰撞
                curr_collision = self._check_collision(new_pos, i)
                if curr_collision > 0:
                    collision_occurred = True
                    collision_type = curr_collision
                    rewards.append(-100.0)  # 碰撞惩罚
                else:
                    rewards.append(0.0)  # 巷道阶段不给奖励

                new_positions.append(new_pos)

            else:
                # RL阶段：使用强化学习控制
                action = actions[i]
                forward_speed = action[0] * self.grid_size / 10.0
                turn_rate = action[1] * (np.pi / 2)

                # 更新朝向和位置
                self.agent_directions[i] += turn_rate * self.interval
                self.agent_directions[i] %= 2 * np.pi

                forward_distance = forward_speed * self.interval
                new_pos = self.agent_positions[i] + np.array([
                    np.cos(self.agent_directions[i]),
                    np.sin(self.agent_directions[i])
                ]) * forward_distance

                # 确保不超出边界
                new_pos = np.clip(new_pos, 0, self.grid_size)

                # 检查碰撞
                curr_collision = self._check_collision(new_pos, i)
                if curr_collision > 0:
                    collision_occurred = True
                    collision_type = curr_collision
                    rewards.append(-100.0)  # 碰撞惩罚
                else:
                    # ==========================================================
                    # 【修改开始】使用您提供的新的奖励计算逻辑
                    # ==========================================================
                    target_pos = self.goal_positions[i]  # 使用 goal_positions 代替未定义的 switching_points
                    current_dist = np.linalg.norm(new_pos - target_pos)
                    prev_dist = np.linalg.norm(self.agent_positions[i] - target_pos)
                    progress = prev_dist - current_dist  # 计算向目标前进的距离

                    if progress > 0.005:  # 如果有效前进了
                        # 1. 基础的距离奖励
                        reward = progress * 20
                        # 2. 额外的“干得好”生存奖励
                        reward += 1.0
                    else:
                        # 如果停滞不前或后退，给予惩罚
                        reward = -3.0
                    # ==========================================================
                    # 【修改结束】
                    # ==========================================================

                    # 2. A*路径引导奖励 (继续累加)
                    reward_astar = 0.0
                    path = self.astar_paths[i]
                    if path is not None and len(path) > 1:
                        min_dist_to_path = float('inf')
                        for point in path:
                            dist = np.linalg.norm(new_pos - point)
                            min_dist_to_path = min(min_dist_to_path, dist)

                        if min_dist_to_path < 1.0:
                            reward_astar = 2.0
                        elif min_dist_to_path > 1.5:
                            reward_astar = -min_dist_to_path * 1.2
                    reward += reward_astar

                    # 3. 到达目标奖励 (继续累加)
                    reward_goal = 0.0
                    # 注意：这里的 current_dist 已经被计算过了，可以直接复用
                    if current_dist < 0.1:
                        reward_goal = (200.0 + (self.max_steps - self.current_step) / self.max_steps * 200.0)
                        self.agent_reached_goal_flag[i] = True
                    reward += reward_goal

                    # 5. 速度变化惩罚 (继续累加)
                    previous_action = self.previous_actions[i]
                    speed_change = abs(action[0] - previous_action[0])
                    reward_speed_change = -speed_change * 1.0
                    reward += reward_speed_change

                    # 6. 角速度惩罚 (继续累加)
                    angular_penalty = abs(action[1])
                    reward_angular = -angular_penalty * 1.5
                    reward += reward_angular

                    # 7.时间步惩罚 (从新的奖励逻辑中移除，统一在这里累加)
                    # 您新的逻辑中已经包含了生存奖励和停滞惩罚，可以考虑是否还需要额外的时间惩罚
                    # 这里我暂时注释掉原来的时间惩罚，以避免双重惩罚。您可以根据需要取消注释。
                    reward_time = -1.5
                    reward += reward_time

                    rewards.append(reward)

                new_positions.append(new_pos)

        # 【修改】保存当前位置和动作，用于下一步计算
        self.previous_positions = [pos.copy() for pos in self.agent_positions]
        self.agent_positions = new_positions
        self.previous_actions = [a.copy() for a in actions]

        # 更新轨迹
        for i in range(self.n_agents):
            self.trajectories[i].append(self.agent_positions[i].copy())

        # 获取新的观察
        observations = self._get_observations()

        # 判断是否结束
        done = collision_occurred or all(self.agent_reached_goal_flag) or self.current_step >= self.max_steps
        truncated = [False] * self.n_agents
        info = {
            "collision": collision_occurred,
            "collision_type": collision_type
        }

        return observations, rewards, [done] * self.n_agents, truncated, info

    def _get_observations(self):
        """获取每个智能体的观察"""
        observations = []

        for i in range(self.n_agents):
            # 初始化观察列表
            obs = []

            # 1. 自身位置 (2维)
            obs.extend(self.agent_positions[i])

            # 2. 目标位置 (2维)
            obs.extend(self.goal_positions[i])

            # 3. 其他智能体信息 (每个智能体4维)
            for j in range(self.n_agents):
                if i != j:
                    # 位置 (2维)
                    obs.extend(self.agent_positions[j])
                    # 速度 (2维)
                    velocity = (self.agent_positions[j] - self.previous_positions[j]) / self.interval
                    obs.extend(velocity)

            # 4. A*路径信息 (每个路标点3维)
            path = self.astar_paths[i]
            waypoints = sample_waypoints(path, self.n_waypoints)

            for waypoint in waypoints:
                # 相对位置 (2维)
                relative_pos = waypoint - self.agent_positions[i]
                obs.extend(relative_pos)
                # 距离 (1维)
                distance = np.linalg.norm(relative_pos)
                obs.append(distance)

            # 确保观察维度正确
            assert len(obs) == self.observation_space.shape[0], \
                f"Observation dimension mismatch. Expected {self.observation_space.shape[0]}, got {len(obs)}"

            observations.append(np.array(obs, dtype=np.float32))

        return observations

    def render(self, mode='human', astar_paths=None):
        """
        渲染环境，支持 'human' 和 'rgb_array' 两种模式
        """
        if mode == 'human':
            # Human mode: 交互式显示
            # 检查窗口是否存在，如果不存在或已关闭则重新创建
            if self.fig_human is None or not plt.fignum_exists(self.fig_human.number):
                self.fig_human, self.ax_human = plt.subplots(figsize=(8, 8), dpi=100)
                plt.ion()

            ax = self.ax_human
            fig = self.fig_human
            ax.clear()

        elif mode == 'rgb_array':
            # RGB array mode: 渲染到后台缓冲区
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

        else:
            raise ValueError(f"Unsupported render mode: {mode}")

        # --- 通用绘图逻辑 ---
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        ax.grid(False)
        agent_colors = ['blue', 'red', 'green']

        # 绘制起点和终点
        for i, (start, goal) in enumerate(zip(self.start_positions, self.goal_positions)):
            ax.add_patch(patches.Rectangle((start[0] - 0.5, start[1] - 0.5), 1, 1,
                                           facecolor='lightgreen', alpha=0.5))
            ax.text(start[0], start[1], f'S{i + 1}', ha='center', va='center')
            ax.add_patch(patches.Rectangle((goal[0] - 0.5, goal[1] - 0.5), 1, 1,
                                           facecolor='lightcoral', alpha=0.5))
            ax.text(goal[0], goal[1], f'G{i + 1}', ha='center', va='center')

        # 绘制线段障碍物(边界)
        if hasattr(self, 'obstacle_lines') and self.obstacle_lines:
            for line_start, line_end in self.obstacle_lines:
                ax.plot(
                    [line_start[0], line_end[0]],
                    [line_start[1], line_end[1]],
                    color='black',
                    linewidth=2
                )

        #     ###绘制A*路径（只在RL阶段显示）
        # for i, path in enumerate(self.astar_paths):
        #     if self.agent_phases[i] == 'rl' and path is not None and isinstance(path, (list, np.ndarray)) and len(
        #             path) > 1:
        #         path_arr = np.array(path)
        #         ax.plot(path_arr[:, 0], path_arr[:, 1], '-',
        #                 color=agent_colors[i], alpha=0.15, linewidth=2)
        #         waypoints = sample_waypoints(path, self.n_waypoints)
        #         waypoints_arr = np.array(waypoints)
        #         ax.scatter(waypoints_arr[:, 0], waypoints_arr[:, 1],
        #                    color=agent_colors[i], alpha=0.5, s=50)

        # 绘制智能体轨迹
        for i, trajectory in enumerate(self.trajectories):
            if len(trajectory) > 1:
                trajectory_arr = np.array(trajectory)
                ax.plot(trajectory_arr[:, 0], trajectory_arr[:, 1], '-',
                        color=agent_colors[i], alpha=0.5, linewidth=1)

        # 绘制智能体
        for i, pos in enumerate(self.agent_positions):
            circle = plt.Circle((pos[0], pos[1]), self.agent_radius,
                                color=agent_colors[i % len(agent_colors)])
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], f'A{i + 1}', ha='center', va='center', color='white')

        # 设置标题
        phase_text = [f"A{i + 1}:{phase}" for i, phase in enumerate(self.agent_phases)]
        ax.set_title(f'  {", ".join(phase_text)}')

        # --- 根据模式完成后续操作 ---
        if mode == 'human':
            # 更新交互式窗口
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
            return None

        elif mode == 'rgb_array':
            # 将画布转换为RGB数组
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())
            plt.close(fig)  # 关闭图形，防止显示并释放内存
            return img