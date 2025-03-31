# SmartTrader V5 (完整版)

AI 強化學習交易機器人，支援多策略（PPO + A2C）、TP/SL 模型、看空判斷、自動推播與每日績效報表！

## ✅ 主要功能
- ✅ 看多/看空決策（支援空單）
- ✅ PPO + A2C 策略同時訓練，自動選擇最佳策略推播
- ✅ TP/SL 回歸模型（風控最佳化）
- ✅ 每日 23:59 自動推播資金績效圖與總結
- ✅ 支援 Binance 即時訓練 + 歷史模擬
- ✅ 可部署於本地或 Render 雲端

## 🚀 部署方式

### 本地執行
```bash
pip install -r requirements.txt
python main.py
```

### Render 雲端部署
1. 推送整包到 GitHub
2. 登入 render.com → New Web Service
3. 使用 `render.yaml` 自動部署
4. 設定環境變數：
   - binance_api_key
   - binance_api_secret
   - telegram_bot_token
   - telegram_chat_id

### 每日推播
- 每天會在 23:59 自動推播資金曲線圖與策略績效報表至 Telegram。

Enjoy your Smart Trading! 💹🤖
