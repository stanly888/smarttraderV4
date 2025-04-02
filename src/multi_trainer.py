
import torch
from torch.distributions import Categorical
from models import PPOPolicy, TP_SL_Model
from env import TradingEnv
from notifications import send_strategy_signal
from datetime import datetime
class MultiStrategyTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPOPolicy().to(self.device)
        self.tp_sl_model = TP_SL_Model().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config["lr_policy"])
        self.tp_sl_optimizer = torch.optim.Adam(self.tp_sl_model.parameters(), lr=config["lr_tp_sl"])

    def train_all(self, close, high, low, volume, is_morning):
        env = TradingEnv(close, high, low, volume, config=self.config)
        state = env.reset()
        done = False
        states, actions, rewards, log_probs, leverages = [], [], [], [], []

        while not done:
            state_tensor = torch.FloatTensor(state).to(self.device)
            probs, leverage = self.policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action.item(), leverage.item())
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            leverages.append(leverage)

            state = next_state

        # 訓練 PPO
        returns = self.compute_returns(rewards)
        self.update_policy(states, actions, log_probs, returns)

        # 訓練 TP/SL 模型
        self.train_tp_sl(states)

        # 推播訊號
        direction = ['觀望', '做多', '做空'][actions[-1]]
        confidence = float(probs[action].item()) * 100
        final_leverage = round(leverages[-1].item(), 2)
        tp_sl = self.tp_sl_model(state_tensor).detach().cpu().numpy()
        result = {
            "symbol": "BTC/USDT",
            "direction": direction,
            "confidence": round(confidence, 2),
            "leverage": final_leverage,
            "tp": round(tp_sl[0], 4),
            "sl": round(tp_sl[1], 4),
            "model": "PPO",
            "is_morning": is_morning,
            "capital": round(env.capital, 2)
        }
        send_strategy_signal(result, self.config)
        return result

    def compute_returns(self, rewards):
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + self.config["gamma"] * G
            returns.insert(0, G)
        return torch.tensor(returns).to(self.device)

    def update_policy(self, states, actions, log_probs, returns):
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        new_probs, _ = self.policy(states)
        dist = Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)
        ratios = torch.exp(new_log_probs - log_probs)
        surr1 = ratios * returns
        surr2 = torch.clamp(ratios, 1 - self.config["eps_clip"], 1 + self.config["eps_clip"]) * returns
        loss = -torch.min(surr1, surr2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_tp_sl(self, states):
        states_tensor = torch.stack(states).to(self.device)
        targets = torch.ones((len(states), 2)).to(self.device)
        preds = self.tp_sl_model(states_tensor)
        loss = torch.nn.functional.mse_loss(preds, targets)
        self.tp_sl_optimizer.zero_grad()
        loss.backward()
        self.tp_sl_optimizer.step()
