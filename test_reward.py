import numpy as np
import pandas as pd
from env import MultiUAVEnv, Config


def test_reward_distribution(steps=10000):
    print("=" * 70)
    print(f"🕵️ 开始进行 {steps} 步随机策略奖励极限压测...")
    print("=" * 70)

    cfg = Config()
    env = MultiUAVEnv(cfg)
    env.reset()

    records = []

    for _ in range(steps):
        # 使用均匀分布的随机动作进行环境探索
        actions = env.action_space.sample()
        next_obs, rewards, term, trunc, info = env.step(actions)

        # 提取 agent_0 的奖励组件（由于环境同构，取其一即可代表整体分布）
        agent_info = info['agent_0'].copy()

        # 补充记录经过 numpy clip 截断后的最终单步总奖励
        agent_info['total_reward'] = rewards[0]

        # 我们只关心奖励组件，剔除掉 z_pos 等物理状态信息
        reward_data = {k: v for k, v in agent_info.items() if k.startswith('r_') or k == 'total_reward'}
        records.append(reward_data)

        if term or trunc:
            env.reset()

    # 转换为 Pandas DataFrame 开启统计引擎
    df = pd.DataFrame(records)

    # 计算核心统计学指标
    stats = df.describe().T
    stats['var'] = df.var()

    # 整理输出列的标准顺序
    target_cols = ['mean', 'var', 'min', '25%', '50%', '75%', 'max']
    stats = stats[target_cols]

    print("\n📊 10000步极限测试：端到端奖励组件数学透视表")
    pd.set_option('display.float_format', lambda x: f'{x:9.4f}')
    print(stats.to_string())
    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_reward_distribution(10000)
    print("🎉 压测完成！请结合均值与方差分析奖励体系的健康度。")