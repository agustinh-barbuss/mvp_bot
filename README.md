
# BTC MVP Bot (1H)

Pipeline MVP:
1) Download (ccxt)
2) Build features (technical indicators)
3) Backtest (with costs)
4) Temporal validation
5) Sentiment filter (Fear & Greed)
6) Walk-forward
7) Paper trading


## Quickstart (Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\00_download.py