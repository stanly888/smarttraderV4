
def calculate_reward(trade_result):
    reward = 0
    if trade_result["hit_tp"]:
        reward += 1
    if trade_result["hit_sl"]:
        reward -= 1.5
    return reward
