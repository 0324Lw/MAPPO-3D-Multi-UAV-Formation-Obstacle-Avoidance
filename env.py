import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


# ==========================================
# 1. Config 配置类 (全量参数化)
# ==========================================
class Config:
    def __init__(self):
        # 物理空间与时间参数
        self.map_size = 50.0  # X, Y 范围 [0, 50]
        self.max_z = 2.0  # Z 范围 [0, 2]
        self.dt = 0.1  # 仿真步长 (秒)
        self.max_steps = 300  # 最大步数

        # 实体参数
        self.num_agents = 3
        self.num_obstacles = 14  # 障碍物总数
        self.obs_radius = 1.5  # 障碍物半径/半边长
        self.obs_clearance = 4.0  # 障碍物之间的最小通行通道宽度 (米)
        self.safe_zone_radius = 3.0  # 起终点安全区半径
        self.min_start_goal_dist = 50.0  # 起终点最小距离约束
        self.goal_radius = 5.0  # 终点判定半径

        # 无人机运动学边界与编队参数
        self.v_max = 3.0  # 最大线速度 (m/s)
        self.omega_max = np.pi / 2  # 最大偏航角速度 (rad/s)
        self.vz_max = 1.0  # 最大垂向速度 (m/s)

        self.uav_radius = 0.3  # 无人机物理碰撞半径
        self.formation_size = 3.0  # 初始编队边长
        self.z_targets = [1.8, 1.2, 1.2]  # U1, U2, U3 的绝对巡航目标高度
        self.social_dist = 2.0  # 防扎堆安全社交距离
        self.form_limit = 6.0  # 弹性编队形变容忍上限 (超过则线性扣分)

        # 传感器参数 (LiDAR)
        self.lidar_rays = 21  # 提升至 21 根，覆盖更密集的死角
        self.lidar_fov = np.pi  # 视场角 180 度
        self.lidar_range = 10.0  # 最大探测距离
        self.lidar_margin = 0.2  # 雷达探测膨胀系数，提前感知尖角

        # 连续奖励函数系数 (端到端)
        self.r_step = -0.02  # 步数生存惩罚
        self.c_alt = 0.05  # 偏离目标高度惩罚系数
        self.c_rep = 0.5  # 防扎堆斥力惩罚系数
        self.c_form = 0.05  # 弹性散架惩罚系数
        self.c_prog = 1.5  # 重心推进奖励系数
        self.c_smooth = 0.015 # 动作平滑惩罚系数

        # 终止状态奖励 (严格遵守 [-2, 2] 单步截断，将终局大奖等比例缩放)
        self.r_crash = -2.0  # 碰撞/越界/追尾的终止惩罚
        self.r_success = 2.0  # 完美编队通关大奖
        self.r_fail_arrive = -1.0  # 散架通关苟活惩罚


# ==========================================
# 2. Env 环境类
# ==========================================
class MultiUAVEnv(gym.Env):
    def __init__(self, cfg=Config()):
        super(MultiUAVEnv, self).__init__()
        self.cfg = cfg

        # 动作空间：[v, omega, vz] -> 归一化 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.num_agents, 3), dtype=np.float32)

        # 状态空间维度：Ego(5) + Share(2*3) + Goal(2) + LiDAR(21) = 34 维
        obs_dim = 5 + 6 + 2 + self.cfg.lidar_rays
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.num_agents, obs_dim), dtype=np.float32)

        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 1. 生成起终点
        while True:
            sx, sy = np.random.uniform(5, 10), np.random.uniform(5, 10)
            if np.random.rand() > 0.5: sx = self.cfg.map_size - sx
            if np.random.rand() > 0.5: sy = self.cfg.map_size - sy
            self.start_pos = np.array([sx, sy, 0.0])

            gx, gy = np.random.uniform(0, self.cfg.map_size), np.random.uniform(0, self.cfg.map_size)
            self.goal_pos = np.array([gx, gy, 0.0])

            if np.linalg.norm(self.start_pos[0:2] - self.goal_pos[0:2]) >= self.cfg.min_start_goal_dist:
                break

        # 2. 生成静态障碍物 (引入 7.0m 的最小间距保证通道畅通)
        self.obstacles = []
        min_obs_dist = 2 * self.cfg.obs_radius + self.cfg.obs_clearance

        for _ in range(self.cfg.num_obstacles):
            while True:
                margin = self.cfg.obs_radius + 1.0
                ox, oy = np.random.uniform(margin, self.cfg.map_size - margin), np.random.uniform(margin,
                                                                                                  self.cfg.map_size - margin)
                pos = np.array([ox, oy])

                if np.linalg.norm(pos - self.start_pos[0:2]) > (self.cfg.formation_size + 3.0) and \
                        np.linalg.norm(pos - self.goal_pos[0:2]) > (self.cfg.goal_radius + 2.0):

                    clear = True
                    for o in self.obstacles:
                        if np.linalg.norm(pos - o['pos']) < min_obs_dist:
                            clear = False;
                            break
                    if clear:
                        self.obstacles.append({'type': 'cyl' if np.random.rand() > 0.5 else 'box', 'pos': pos})
                        break

        # 3. 初始化无人机 (【核心修复】全部从 Z=0 地面起飞)
        vec_s2g = self.goal_pos[0:2] - self.start_pos[0:2]
        dir_forward = vec_s2g / (np.linalg.norm(vec_s2g) + 1e-6)
        dir_side = np.array([-dir_forward[1], dir_forward[0]])
        init_yaw = np.arctan2(dir_forward[1], dir_forward[0])

        self.states = np.zeros((self.cfg.num_agents, 6))  # [x, y, z, v, yaw, vz]
        f_s = self.cfg.formation_size

        # U1 (领航机)
        self.states[0, 0:2] = self.start_pos[0:2]
        # U2 (左僚机)
        self.states[1, 0:2] = self.start_pos[0:2] - dir_forward * (f_s * 0.866) + dir_side * (f_s / 2)
        # U3 (右僚机)
        self.states[2, 0:2] = self.start_pos[0:2] - dir_forward * (f_s * 0.866) - dir_side * (f_s / 2)

        # 所有无人机初始 Z 坐标强制为 0
        self.states[:, 2] = 0.0
        self.states[:, 4] = init_yaw
        self.prev_actions = np.zeros((self.cfg.num_agents, 3))
        self.prev_com_dist = np.linalg.norm(np.mean(self.states[:, 0:3], axis=0)[0:2] - self.goal_pos[0:2])
        return self._get_obs(), {}

    def step(self, actions):
        self.current_step += 1
        rewards = np.zeros(self.cfg.num_agents)
        info = {f"agent_{i}": {} for i in range(self.cfg.num_agents)}

        # --- 1. 运动学更新 ---
        for i in range(self.cfg.num_agents):
            act_v = (np.clip(actions[i, 0], -1.0, 1.0) + 1.0) / 2.0 * self.cfg.v_max
            act_w = np.clip(actions[i, 1], -1.0, 1.0) * self.cfg.omega_max
            act_vz = np.clip(actions[i, 2], -1.0, 1.0) * self.cfg.vz_max

            self.states[i, 3] = act_v
            self.states[i, 5] = act_vz
            self.states[i, 4] = (self.states[i, 4] + act_w * self.cfg.dt + np.pi) % (2 * np.pi) - np.pi

            self.states[i, 0] += act_v * np.cos(self.states[i, 4]) * self.cfg.dt
            self.states[i, 1] += act_v * np.sin(self.states[i, 4]) * self.cfg.dt
            self.states[i, 2] = np.clip(self.states[i, 2] + act_vz * self.cfg.dt, 0.0, self.cfg.max_z)

        # --- 2. 状态与指标计算 ---
        com_pos = np.mean(self.states[:, 0:3], axis=0)
        curr_com_dist = np.linalg.norm(com_pos[0:2] - self.goal_pos[0:2])
        delta_prog = self.prev_com_dist - curr_com_dist
        self.prev_com_dist = curr_com_dist

        # 计算编队形变
        d01 = np.linalg.norm(self.states[0, 0:3] - self.states[1, 0:3])
        d12 = np.linalg.norm(self.states[1, 0:3] - self.states[2, 0:3])
        d20 = np.linalg.norm(self.states[2, 0:3] - self.states[0, 0:3])
        max_form_dist = max(d01, d12, d20)

        # --- 3. 严格的终止碰撞检测 ---
        is_oob = np.zeros(self.cfg.num_agents, dtype=bool)
        is_obs_col = np.zeros(self.cfg.num_agents, dtype=bool)
        is_uav_col = np.zeros(self.cfg.num_agents, dtype=bool)

        for i in range(self.cfg.num_agents):
            x, y, z = self.states[i, 0:3]
            # 越界检测
            if x < 0 or x > self.cfg.map_size or y < 0 or y > self.cfg.map_size:
                is_oob[i] = True

            # 追尾检测 (硬碰撞)
            for j in range(self.cfg.num_agents):
                if i != j and np.linalg.norm(self.states[i, 0:3] - self.states[j, 0:3]) < self.cfg.uav_radius * 2:
                    is_uav_col[i] = True

            # 障碍物检测
            for obs in self.obstacles:
                if obs['type'] == 'cyl':
                    if np.linalg.norm([x - obs['pos'][0], y - obs['pos'][1]]) < (
                            self.cfg.obs_radius + self.cfg.uav_radius):
                        is_obs_col[i] = True
                else:  # box
                    if abs(x - obs['pos'][0]) < (self.cfg.obs_radius + self.cfg.uav_radius) and \
                            abs(y - obs['pos'][1]) < (self.cfg.obs_radius + self.cfg.uav_radius):
                        is_obs_col[i] = True

        agent_crashed = is_oob | is_obs_col | is_uav_col
        any_crash = np.any(agent_crashed)
        is_success = curr_com_dist <= self.cfg.goal_radius

        # 终止条件：要么撞了，要么到了
        terminated = any_crash or is_success
        truncated = self.current_step >= self.cfg.max_steps

        # --- 4. 端到端连续奖励核算 ---
        for i in range(self.cfg.num_agents):
            r_step = self.cfg.r_step

            # 1. 高度锁定奖励 (惩罚偏差)
            r_alt = -self.cfg.c_alt * abs(self.states[i, 2] - self.cfg.z_targets[i])

            # 2. 防扎堆斥力奖励 (只看软接触，不触发Done，但扣分极狠)
            r_rep = 0.0
            for j in range(self.cfg.num_agents):
                if i != j:
                    dist_ij = np.linalg.norm(self.states[i, 0:3] - self.states[j, 0:3])
                    if dist_ij < self.cfg.social_dist:
                        r_rep -= self.cfg.c_rep * (self.cfg.social_dist - dist_ij) ** 2

            # 3. 弹性编队奖励 (超过 6m 才开始扣分)
            r_form = 0.0
            if max_form_dist > self.cfg.form_limit:
                r_form = -self.cfg.c_form * (max_form_dist - self.cfg.form_limit)

            # 4. 推进奖励
            r_prog = self.cfg.c_prog * delta_prog

            r_smooth = -self.cfg.c_smooth * np.sum((actions[i] - self.prev_actions[i]) ** 2)

            # 5. 终局结算
            r_term = 0.0
            if any_crash:
                r_term = self.cfg.r_crash  # 一损俱损
            elif is_success:
                if max_form_dist <= self.cfg.form_limit:
                    r_term = self.cfg.r_success
                else:
                    r_term = self.cfg.r_fail_arrive  # 抛弃队友独自到达的重罚

            total_r = r_step + r_alt + r_rep + r_form + r_prog + r_term + r_smooth

            # 【严格限幅】保证单步返回值在 [-2.0, 2.0]
            rewards[i] = np.clip(total_r, -2.0, 2.0)

            info[f"agent_{i}"] = {
                "r_step": r_step, "r_alt": r_alt, "r_rep": r_rep,
                "r_form": r_form, "r_prog": r_prog, "r_term": r_term,"r_smooth": r_smooth,
                "z_pos": self.states[i, 2]
            }
        self.prev_actions = np.copy(actions)

        return self._get_obs(), rewards, terminated, truncated, info

    def _get_obs(self):
        obs = np.zeros((self.cfg.num_agents, self.observation_space.shape[1]), dtype=np.float32)
        com_pos = np.mean(self.states[:, 0:3], axis=0)

        for i in range(self.cfg.num_agents):
            idx = 0
            # Ego
            obs[i, idx:idx + 3] = self.states[i, 0:3] / np.array([self.cfg.map_size, self.cfg.map_size, self.cfg.max_z])
            obs[i, idx + 3] = self.states[i, 3] / self.cfg.v_max
            obs[i, idx + 4] = self.states[i, 4] / np.pi
            idx += 5

            # Share (相对队友)
            for j in range(self.cfg.num_agents):
                if i != j:
                    rel_pos = self.states[j, 0:3] - self.states[i, 0:3]
                    obs[i, idx:idx + 3] = np.clip(rel_pos / 10.0, -1.0, 1.0)
                    idx += 3

            # Goal (相对重心)
            vec_goal = self.goal_pos[0:2] - com_pos[0:2]
            dist_goal = np.linalg.norm(vec_goal)
            angle_goal = np.arctan2(vec_goal[1], vec_goal[0])
            angle_diff = (angle_goal - self.states[i, 4] + np.pi) % (2 * np.pi) - np.pi
            obs[i, idx] = np.clip(dist_goal / self.cfg.map_size, 0.0, 1.0)
            obs[i, idx + 1] = angle_diff / np.pi
            idx += 2

            # LiDAR (21 根射线，带安全膨胀)
            rays = np.linspace(-self.cfg.lidar_fov / 2, self.cfg.lidar_fov / 2, self.cfg.lidar_rays)
            lidar_dists = np.ones(self.cfg.lidar_rays)
            px, py, yaw = self.states[i, 0], self.states[i, 1], self.states[i, 4]

            for r_idx, angle in enumerate(rays):
                ray_dir = np.array([np.cos(yaw + angle), np.sin(yaw + angle)])
                min_t = self.cfg.lidar_range

                for obs_obj in self.obstacles:
                    vec_to_obs = obs_obj['pos'] - np.array([px, py])
                    t = np.dot(vec_to_obs, ray_dir)
                    if 0 < t < self.cfg.lidar_range:
                        perp_dist = np.linalg.norm(vec_to_obs - t * ray_dir)
                        # 雷达膨胀：让网络把障碍物看得比实际大 0.2m，避免擦角
                        col_r = (self.cfg.obs_radius + self.cfg.lidar_margin) if obs_obj['type'] == 'cyl' else (
                                    self.cfg.obs_radius * 1.414 + self.cfg.lidar_margin)

                        if perp_dist < col_r:
                            actual_dist = max(0.0, t - np.sqrt(col_r ** 2 - perp_dist ** 2))
                            if actual_dist < min_t: min_t = actual_dist
                lidar_dists[r_idx] = min_t / self.cfg.lidar_range
            obs[i, idx:idx + self.cfg.lidar_rays] = lidar_dists
        return obs


# ==========================================
# 3. Plotter 绘图类
# ==========================================
class Plotter:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_training_curves(self, rewards_history, lengths_history, save_path="training_curves.png"):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color = 'tab:blue'
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Episodic Reward', color=color)
        ax1.plot(rewards_history, color=color, alpha=0.4, label='Reward')
        ax1.tick_params(axis='y', labelcolor=color)

        if len(rewards_history) >= 20:
            smoothed = np.convolve(rewards_history, np.ones(20) / 20, mode='valid')
            ax1.plot(range(19, len(rewards_history)), smoothed, color='darkblue', linewidth=2)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Episode Length', color=color)
        ax2.plot(lengths_history, color=color, alpha=0.3, label='Length')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Multi-UAV Formation Navigation Training')
        plt.savefig(save_path, dpi=300)
        plt.close()