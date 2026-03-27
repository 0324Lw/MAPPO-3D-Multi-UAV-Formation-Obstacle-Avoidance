import os
import torch
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 必须使用 Agg 后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle
import imageio
import torch.nn as nn
from env import MultiUAVEnv, Config


# ==========================================
# 1. 对齐训练时的 Actor 结构
# ==========================================
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim)
        )
        # 兼容性参数：加载时会被填充，但推理时不使用
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_action(self, x):
        return self.network(x)


# ==========================================
# 2. 录制与可视化核心逻辑
# ==========================================
def evaluate_and_record(model_path="models/actor_final.pth", num_episodes=10):
    print("=" * 80)
    print(f"🎬 正在启动 [3D起飞+避障] 联合视图录制器...")
    print("=" * 80)

    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config()
    env = MultiUAVEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化并加载模型
    obs_dim = env.observation_space.shape[1]
    act_dim = env.action_space.shape[1]
    actor = Actor(obs_dim, act_dim).to(device)

    try:
        sd = torch.load(model_path, map_location=device)
        # 自动处理 'net' 到 'network' 的映射
        new_sd = {k.replace('net.', 'network.'): v for k, v in sd.items()}
        actor.load_state_dict(new_sd, strict=False)
        print(f"✅ 成功加载模型: {model_path}")
    except:
        print(f"❌ 未能找到模型文件，请确保路径正确。")
        return

    actor.eval()
    uav_colors = ['#1f77b4', '#9467bd', '#ff7f0e']  # 蓝、紫、橙

    for ep in range(num_episodes):
        obs, _ = env.reset()
        frames = []
        trajs = [[] for _ in range(cfg.num_agents)]
        done = False
        step = 0

        # 预设画布
        fig = plt.figure(figsize=(20, 7))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1])
        ax3d = fig.add_subplot(gs[0], projection='3d')
        ax_xy = fig.add_subplot(gs[1])
        ax_xoz = fig.add_subplot(gs[2])

        print(f"[*] 录制中 Episode {ep + 1}...")

        while not done:
            with torch.no_grad():
                a = actor.get_action(torch.FloatTensor(obs).to(device)).cpu().numpy()

            obs, rewards, term, trunc, info = env.step(a)
            done = term or trunc
            step += 1

            # 记录轨迹
            for i in range(cfg.num_agents):
                trajs[i].append(env.states[i, 0:3].copy())

            # --- 绘图逻辑 ---
            for ax in [ax3d, ax_xy, ax_xoz]: ax.cla()

            # 1. 3D Global View
            ax3d.view_init(elev=20, azim=45)
            ax3d.set_zlim(0, 2.2);
            ax3d.set_title("3D Formation Flight")
            ax3d.scatter(*env.start_pos, c='b', marker='x');
            ax3d.scatter(*env.goal_pos, c='g', marker='*')

            # 2. XY Top-Down (带社交距离圈)
            ax_xy.set_xlim(0, 50);
            ax_xy.set_ylim(0, 50);
            ax_xy.set_aspect('equal')
            ax_xy.set_title("XY Obstacle Avoidance")
            for obj in env.obstacles:
                r = cfg.obs_radius
                color = 'red' if obj['type'] == 'cyl' else 'darkred'
                if obj['type'] == 'cyl':
                    ax_xy.add_patch(Circle(obj['pos'], r, color=color, alpha=0.3))
                else:
                    ax_xy.add_patch(Rectangle(obj['pos'] - r, r * 2, r * 2, color=color, alpha=0.3))

            # 3. XOZ Side-View (看高度锁定)
            ax_xoz.set_xlim(0, 50);
            ax_xoz.set_ylim(-0.1, 2.2);
            ax_xoz.set_title("XOZ Altitude Lock")
            for i, target_z in enumerate(cfg.z_targets):
                ax_xoz.axhline(target_z, color=uav_colors[i], linestyle='--', alpha=0.3)

            # 绘制无人机实体与轨迹
            for i in range(cfg.num_agents):
                curr = env.states[i, 0:3]
                t = np.array(trajs[i])

                # 3D
                ax3d.scatter(curr[0], curr[1], curr[2], c=uav_colors[i], s=40, edgecolors='k')
                ax3d.plot(t[:, 0], t[:, 1], t[:, 2], c=uav_colors[i], alpha=0.6)

                # XY
                ax_xy.plot(curr[0], curr[1], 'o', color=uav_colors[i], markersize=8, markeredgecolor='k')
                ax_xy.plot(t[:, 0], t[:, 1], '-', color=uav_colors[i], alpha=0.4)
                # 绘制防扎堆社交圈 (D=2.0m)
                ax_xy.add_patch(
                    Circle(curr[0:2], cfg.social_dist / 2, color=uav_colors[i], fill=False, linestyle=':', alpha=0.5))

                # XOZ
                ax_xoz.plot(curr[0], curr[2], 'o', color=uav_colors[i], markersize=8, markeredgecolor='k')
                ax_xoz.plot(t[:, 0], t[:, 2], '-', color=uav_colors[i], alpha=0.4)

            fig.tight_layout()
            fig.canvas.draw()
            # 核心修正：使用 .copy() 确保帧独立
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            frames.append(img)

        plt.close(fig)

        # 保存 GIF
        tag = "SUCCESS" if info['agent_0']['r_term'] > 0 else "FAILED"
        path = f"{out_dir}/joint_view_ep{ep + 1}_{tag}.gif"
        imageio.mimsave(path, frames, fps=15)
        print(f"   ↳ 已保存: {path}")


if __name__ == "__main__":
    evaluate_and_record(model_path="models/actor_1001472.pth", num_episodes=10)