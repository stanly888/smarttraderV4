# SmartTrader V4

An AI-powered trading bot using PPO and Bollinger Bands for TP/SL.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Update `config/config.json` with your Binance API credentials.

Run the bot:
```bash
python main.py
```

### Features
- Morning: Uses Binance data for real-time trading.
- Afternoon: Trains on random historical K-line data.
- Telegram notifications for strategies and daily summaries.

### Deployment
GitHub: Push this repo to your GitHub account.

Render: Use the `render.yaml` file to deploy.
