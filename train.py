import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from env import MultiUAVEnv, Config, Plotter


# ==========================================
# 1. 神经网络与稳定性初始化
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化：避免梯度消失/爆炸，提高初期探索稳定性"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # 针对 34 维状态，128-128 的隐层容量足够且高效
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 128)), nn.Tanh(),
            # 最后一层 std=0.01 保证初始动作接近 0，防止无人机开局乱飞坠机
            layer_init(nn.Linear(128, act_dim), std=0.01),
        )
        # 独立的可学习标准差参数，用于探索
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_action(self, x, action=None):
        action_mean = self.network(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1)


class Critic(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()
        # Centralized Critic: 输入所有智能体的状态拼接 (34 * 3 = 102 维)
        self.network = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_value(self, x):
        return self.network(x)


# ==========================================
# 2. MAPPO 训练核心管理器
# ==========================================
class MAPPOTrainer:
    def __init__(self):
        self.cfg = Config()
        self.env = MultiUAVEnv(self.cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] 训练设备: {self.device}")

        # 维度提取
        self.num_agents = self.cfg.num_agents
        self.obs_dim = self.env.observation_space.shape[1]
        self.act_dim = self.env.action_space.shape[1]
        self.global_state_dim = self.obs_dim * self.num_agents

        # 实例化网络
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.global_state_dim).to(self.device)

        # 超参数设定
        self.initial_lr = 3e-4
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.initial_lr, eps=1e-5
        )

        # PPO 与 GAE 参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_coef = 0.2
        self.ent_coef = 0.01  # 熵正则化系数 (鼓励探索)
        self.vf_coef = 0.5  # 价值函数损失权重
        self.max_grad_norm = 0.5  # 梯度裁剪防爆炸
        self.update_epochs = 10  # 每次采样的网络更新轮数
        self.batch_size = 2048  # rollout 的步长

        # 记录与持久化
        self.plotter = Plotter()
        self.history_rewards = []
        self.history_lengths = []
        os.makedirs("models", exist_ok=True)

    def train(self, total_timesteps=5000000):
        print("=" * 60)
        print(f"🚀 开始 MAPPO 多无人机编队训练 (目标步数: {total_timesteps})")
        print("=" * 60)

        obs, _ = self.env.reset()
        global_step = 0
        start_time = time.time()

        # 预分配 Tensor 内存空间提升计算效率
        storage_obs = torch.zeros((self.batch_size, self.num_agents, self.obs_dim)).to(self.device)
        storage_actions = torch.zeros((self.batch_size, self.num_agents, self.act_dim)).to(self.device)
        storage_logprobs = torch.zeros((self.batch_size, self.num_agents)).to(self.device)
        storage_rewards = torch.zeros((self.batch_size, self.num_agents)).to(self.device)
        storage_dones = torch.zeros(self.batch_size).to(self.device)
        storage_values = torch.zeros((self.batch_size, 1)).to(self.device)

        ep_rewards, ep_len = 0, 0
        success_history = []  # 记录最近 100 局的成功状态
        total_episodes = 0

        while global_step < total_timesteps:
            # --- 0. 学习率衰减 (Linear LR Scheduler) ---
            frac = 1.0 - (global_step - 1.0) / total_timesteps
            lrnow = frac * self.initial_lr
            self.optimizer.param_groups[0]["lr"] = lrnow

            # --- 1. 数据采集阶段 (Rollout) ---
            for step in range(self.batch_size):
                global_step += 1
                with torch.no_grad():
                    # Actor 推理 (独立状态)
                    torch_obs = torch.FloatTensor(obs).to(self.device)
                    actions, logprobs, _ = self.actor.get_action(torch_obs)

                    # Critic 推理 (全局状态展平)
                    global_state = torch_obs.view(1, -1)
                    value = self.critic.get_value(global_state)

                # 与环境交互
                next_obs, rewards, term, trunc, info = self.env.step(actions.cpu().numpy())
                done = term or trunc

                # 存入缓冲区
                storage_obs[step] = torch_obs
                storage_actions[step] = actions
                storage_logprobs[step] = logprobs
                storage_rewards[step] = torch.FloatTensor(rewards).to(self.device)
                storage_dones[step] = float(done)
                storage_values[step] = value.squeeze()

                obs = next_obs
                ep_rewards += np.mean(rewards)
                ep_len += 1

                if done:
                    total_episodes += 1
                    # 检查是否达成完美的编队成功 (r_term > 0 说明拿到 r_success)
                    is_perfect_success = 1 if info['agent_0']['r_term'] > 0 else 0
                    success_history.append(is_perfect_success)
                    if len(success_history) > 100: success_history.pop(0)  # 只保留最近100局

                    self.history_rewards.append(ep_rewards)
                    self.history_lengths.append(ep_len)

                    obs, _ = self.env.reset()
                    ep_rewards, ep_len = 0, 0

            # --- 2. GAE 优势函数计算 ---
            with torch.no_grad():
                next_global_state = torch.FloatTensor(obs).to(self.device).view(1, -1)
                next_value = self.critic.get_value(next_global_state).squeeze()

                advantages = torch.zeros_like(storage_rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.batch_size)):
                    if t == self.batch_size - 1:
                        nextnonterminal = 1.0 - float(done)
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - storage_dones[t + 1]
                        nextvalues = storage_values[t + 1]

                    # 为每个 Agent 计算 delta
                    delta = storage_rewards[t] + self.gamma * nextvalues.unsqueeze(0) * nextnonterminal - \
                            storage_values[t].unsqueeze(0)
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + storage_values

            # --- 3. 网络更新阶段 (PPO Epochs) ---
            b_obs = storage_obs.reshape(-1, self.obs_dim)
            b_actions = storage_actions.reshape(-1, self.act_dim)
            b_logprobs = storage_logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = storage_values.expand(-1, self.num_agents).reshape(-1)

            total_loss_record = 0.0

            for epoch in range(self.update_epochs):
                _, newlogprob, entropy = self.actor.get_action(b_obs, b_actions)

                # 重建 batch 的 global_state 送入 critic
                reshaped_obs = b_obs.view(-1, self.num_agents * self.obs_dim)
                newvalue = self.critic.get_value(reshaped_obs)
                # 将 critic 的输出对齐到每个 agent
                newvalue = newvalue.repeat_interleave(self.num_agents, dim=0).view(-1)

                logratio = newlogprob - b_logprobs
                ratio = logratio.exp()

                # 优势归一化 (Mini-batch 级别，极大提升稳定性)
                mb_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # 策略损失
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 价值损失
                v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

                # 熵正则化损失
                entropy_loss = entropy.mean()

                # 总体损失
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                total_loss_record += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.max_grad_norm)
                self.optimizer.step()

            # --- 4. 日志打印与模型保存 ---
            avg_loss = total_loss_record / self.update_epochs
            avg_r = np.mean(self.history_rewards[-20:]) if len(self.history_rewards) > 0 else 0
            avg_l = np.mean(self.history_lengths[-20:]) if len(self.history_lengths) > 0 else 0
            recent_success_rate = np.mean(success_history) if len(success_history) > 0 else 0.0
            fps = int(global_step / (time.time() - start_time))

            print(f"Step: {global_step:>8} | Ep: {total_episodes:>5} | "
                  f"Reward: {avg_r:>6.2f} | Len: {avg_l:>5.1f} | "
                  f"Success(100): {recent_success_rate:>5.1%} | Loss: {avg_loss:>6.3f} | FPS: {fps}")

            # 每 50 万步自动保存一次持久化模型
            if global_step % 500000 < self.batch_size:
                torch.save(self.actor.state_dict(), f"models/actor_{global_step}.pth")

        # 训练完全结束，生成趋势图并保存最终模型
        print("\n🎉 训练完成! 正在生成趋势可视化图表...")
        self.plotter.plot_training_curves(self.history_rewards, self.history_lengths)
        torch.save(self.actor.state_dict(), "models/actor_final.pth")
        print("✅ 模型及图表均已保存至工作目录！")


if __name__ == "__main__":
    trainer = MAPPOTrainer()
    # 鉴于问题难度，这里设定总步数为 5,000,000 步
    trainer.train(total_timesteps=5000000)