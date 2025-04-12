from ppo_trainer import train_ppo
from a2c_trainer import train_a2c

def train_model():
    result_ppo = train_ppo()
    result_a2c = train_a2c()
    return max([result_ppo, result_a2c], key=lambda x: x["score"])