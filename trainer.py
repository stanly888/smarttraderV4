    def train(self, episodes=50):
        for ep in range(episodes):
            state = self.env.reset()
            log_probs, old_log_probs, rewards, actions = [], [], [], []
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state)
                with torch.no_grad():
                    old_probs = self.old_policy(state_tensor)
                probs = self.policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                old_log_prob = Categorical(old_probs).log_prob(action)
                state, reward, done = self.env.step(action.item())
                log_probs.append(log_prob)
                old_log_probs.append(old_log_prob)
                rewards.append(reward)
                actions.append(action.item())

            self.old_policy.load_state_dict(self.policy.state_dict())

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)
            log_probs = torch.stack(log_probs)
            old_log_probs = torch.stack(old_log_probs)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * returns
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * returns
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # === 推播訊息組合區 ===
            final_action = actions[-1]
            direction = ['觀望', '做多', '做空'][final_action]

            with torch.no_grad():
                probs = self.policy(torch.FloatTensor(state))
            confidence = round(float(probs[final_action].item()) * 100, 2)

            tp = round(2 + confidence * 0.03, 2)
            sl = round(tp / 3, 2)

            if confidence > 90:
                leverage = 20
            elif confidence > 70:
                leverage = 10
            else:
                leverage = 5

            strategy = {
                'symbol': self.symbol,
                'direction': direction,
                'reason': f'AI PPO 策略（{self.env.mode} 模式）',
                'leverage': leverage,
                'confidence': confidence,
                'tp': tp,
                'sl': sl,
                'model': 'PPO_Strategy'
            }

            send_strategy_signal(strategy)
            log_strategy(strategy, result=round((self.env.capital - 300) / 3, 2))
            print(f"✅ Episode {ep+1} Finished. Capital: {round(self.env.capital, 2)}")

