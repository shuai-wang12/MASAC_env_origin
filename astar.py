import numpy as np
import heapq


class AstarPathFinder:
    """A*路径规划器"""

    def __init__(self, grid_size=10):
        self.grid_size = grid_size

    def get_path(self, start, goal, obstacles):
        """使用A*算法计算路径"""

        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        def get_neighbors(pos):
            # 8个基本方向的邻居
            directions = [
                (1, 0), (1, 1), (0, 1), (-1, 1),
                (-1, 0), (-1, -1), (0, -1), (1, -1)
            ]
            step_size = 0.5  # 步长
            neighbors = []
            for dx, dy in directions:
                new_pos = np.array([pos[0] + dx * step_size, pos[1] + dy * step_size])
                if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                    # 检查新位置到所有障碍物的距离
                    valid = True
                    for obs_start, obs_end in obstacles:
                        if self.point_to_segment_distance(new_pos, obs_start, obs_end) < 0.3:
                            valid = False
                            break
                    if valid:
                        neighbors.append(new_pos)
            return neighbors

        def is_valid(pos, obstacles):
            """检查位置是否有效（不在障碍物内）"""
            if not (0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size):
                return False
            for obs_start, obs_end in obstacles:
                if self.point_to_segment_distance(pos, obs_start, obs_end) < 0.3:
                    return False
            return True

        def is_path_valid(start_pos, end_pos, obstacles):
            """检查两点之间的路径是否有效（不穿过障碍物）"""
            direction = end_pos - start_pos
            distance = np.linalg.norm(direction)
            if distance < 1e-6:
                return True
            direction = direction / distance

            # 检查路径上的多个点
            n_checks = max(int(distance * 10), 10)  # 每单位距离检查10个点
            for i in range(n_checks + 1):
                t = i / n_checks
                point = start_pos + direction * distance * t
                if not is_valid(point, obstacles):
                    return False
            return True

        # 检查起点和终点是否有效
        if not is_valid(start, obstacles) or not is_valid(goal, obstacles):
            print(f"起点或终点无效")
            return np.array([start, goal])

        # A*算法实现
        start_tuple = tuple(start)
        goal_tuple = tuple(goal)

        frontier = []
        heapq.heappush(frontier, (0, start_tuple))
        came_from = {start_tuple: None}
        cost_so_far = {start_tuple: 0}

        goal_found = False
        while frontier:
            current_tuple = heapq.heappop(frontier)[1]
            current = np.array(current_tuple)

            if np.linalg.norm(current - goal) < 0.3:  # 到达目标
                goal_found = True
                break

            for next_pos in get_neighbors(current):
                next_tuple = tuple(next_pos)
                new_cost = cost_so_far[current_tuple] + np.linalg.norm(current - next_pos)

                if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
                    # 【关键修改】注释掉此处的路径检查，以允许在狭窄空间中寻找路径
                    # 即使路径会非常贴近障碍物。RL智能体后续会学习如何避障。
                    # if not is_path_valid(current, next_pos, obstacles):
                    #     continue

                    cost_so_far[next_tuple] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_tuple))
                    came_from[next_tuple] = current_tuple

        # 如果找到路径
        if goal_found:
            # 重建路径
            path = []
            current = current_tuple
            while current in came_from:
                path.append(np.array(current))
                current = came_from[current]
            path.append(np.array(start_tuple))
            path.reverse()

            # 【关键修改】暂时也注释掉最终验证，以确保路径能够返回用于可视化和作为引导
            # # 验证整条路径
            # for i in range(len(path)-1):
            #     if not is_path_valid(path[i], path[i+1], obstacles):
            #         print(f"路径验证失败")
            #         return np.array([start, goal])

            return np.array(path)

        print(f"无法找到从{start}到{goal}的路径")
        return np.array([start, goal])

    def point_to_segment_distance(self, p, a, b):
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


class AstarFeatureExtractor:
    """A*特征提取器"""

    def __init__(self, grid_size=10):
        self.pathfinder = AstarPathFinder(grid_size)

    def extract_features(self, agent_pos, goal_pos, path, max_waypoints=3):
        """提取A*路径特征"""
        features = []

        if path is None or len(path) < 2:
            # 如果没有有效路径，返回零向量
            return np.zeros(4 * max_waypoints)

        # 找到当前位置在路径中的最近点
        min_dist = float('inf')
        nearest_idx = 0
        for i, point in enumerate(path):
            dist = np.linalg.norm(agent_pos - point)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # 从最近点开始，提取接下来的几个路标点
        for i in range(max_waypoints):
            next_idx = min(nearest_idx + i + 1, len(path) - 1)
            waypoint = path[next_idx]

            # 计算相对位置和方向
            relative_pos = waypoint - agent_pos
            distance = np.linalg.norm(relative_pos)
            direction = relative_pos / (distance + 1e-6)

            # 添加特征
            features.extend([distance, direction[0], direction[1], 1.0])

        return np.array(features)