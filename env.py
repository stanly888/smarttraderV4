def calculate_reward(result):
    reward = result["confidence"]
    if result["tp"] > result["sl"]:
        reward += 1
    return reward