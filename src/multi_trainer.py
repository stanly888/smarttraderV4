import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from src.models import PPOPolicy, TP_SL_Model
from src.env import TradingEnv
from src.trade_executor import simulate_trade

class MultiStrategyTrainer:
    def __init__(self, config):
        self.config = config
        self.policy = PPOPolicy()
        self.tp_sl_model = TP_SL_Model()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config["lr_policy"])
        self.tp_sl_optimizer = optim.Adam(self.tp_sl_model.parameters(), lr=config["lr_tp_sl"])
        self.old_policy = PPOPolicy()
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.load_model_if_exists()

    def load_model_if_exists(self):
        try:
            if os.path.exists("ppo_model.pt"):
                self.policy.load_state_dict(torch.load("ppo_model.pt"))
                print("✅ 載入上次訓練的 PPO 模型")
            if os.path.exists("tp_sl_model.pt"):
                self.tp_sl_model.load_state_dict(torch.load("tp_sl_model.pt"))
                print("✅ 載入上次訓練的 TP/SL 模型")
        except Exception as e:
            print(f"❌ 載入模型失敗：{e}")

    def compute_returns(self, rewards):
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + self.config["gamma"] * G
            returns.insert(0, G)
        return torch.tensor(returns)

    def train_all(self, close, high, low, volume):
        env = TradingEnv(close, high, low, is_real_time=True, config=self.config)
        best_strategy = None

        for ep in range(self.config["episodes"]):
            state = env.reset()
            states, log_probs, old_log_probs, rewards, actions = [], [], [], [], []
            leverages, tp_sls = [], []

            done = False
            while not done:
                state_tensor = torch.FloatTensor(state)
                probs, leverage = self.policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                old_probs, _ = self.old_policy(state_tensor)
                old_log_prob = Categorical(old_probs).log_prob(action)

                tp_sl = self.tp_sl_model(state_tensor)

                state, reward, done = env.step(action.item())
                log_probs.append(log_prob)
                old_log_probs.append(old_log_prob)
                rewards.append(reward)
                actions.append(action.item())
                states.append(state_tensor)
                leverages.append(leverage)
                tp_sls.append(tp_sl)

            # Policy 更新
            returns = self.compute_returns(rewards)
            states = torch.stack(states)
            log_probs = torch.stack(log_probs)
            old_log_probs = torch.stack(old_log_probs)

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * returns
            surr2 = torch.clamp(ratios, 1 - self.config["eps_clip"], 1 + self.config["eps_clip"]) * returns
            policy_loss = -torch.min(surr1, surr2).mean()
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            self.old_policy.load_state_dict(self.policy.state_dict())

            # TP/SL 模型更新
            preds = self.tp_sl_model(states)
            targets = torch.tensor([[1.0, 1.0] for _ in range(len(states))], dtype=torch.float32)
            tp_sl_loss = nn.MSELoss()(preds, targets)
            self.tp_sl_optimizer.zero_grad()
            tp_sl_loss.backward()
            self.tp_sl_optimizer.step()

            # 模擬交易
            final_action = actions[-1]
            direction = ["觀望", "做多", "做空"][final_action]
            current_price = env.close[env.index]
            bollinger = env.get_bollinger()
            confidence = round(float(probs[final_action].item()) * 100, 2)
            tp = round(float((bollinger[2] - current_price) * tp_sl[0].item() / current_price), 4)
            sl = round(float((current_price - bollinger[0]) * tp_sl[1].item() / current_price), 4)
            leverage = round(float(leverages[-1].item()), 2)

            result = {
                "timestamp": str(ep),
                "model": "PPO",
                "direction": direction,
                "confidence": confidence,
                "tp": tp,
                "sl": sl,
                "leverage": leverage,
                "symbol": "BTC/USDT"
            }

            # 模擬績效（可擴充）
            result["simulated_pnl"], result["hit_tp"] = simulate_trade(current_price, tp, sl)

            best_strategy = result  # 可擴充為多策略比較後選最強的

        # ✅ 儲存模型
        torch.save(self.policy.state_dict(), "ppo_model.pt")
        torch.save(self.tp_sl_model.state_dict(), "tp_sl_model.pt")
        print("✅ 模型已儲存完畢")

        return best_strategy
