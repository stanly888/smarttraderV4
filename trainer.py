
from models import PPOModel, predict_tp_sl, predict_leverage
from utils import get_market_data
from logger import log_trade

def train_model():
    state = get_market_data()
    model = PPOModel()
    direction, confidence = model.predict(state)
    tp, sl = predict_tp_sl(state)
    leverage = predict_leverage(state)
    result = {
        "direction": direction,
        "confidence": confidence,
        "tp": tp,
        "sl": sl,
        "leverage": leverage
    }
    log_trade(result)
    return result
