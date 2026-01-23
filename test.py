import pandas as pd
p = r"C:\proyectos\mvp_bot\outputs\runs\20260122_150754\trades.csv"
df = pd.read_csv(p)
print("trades:", len(df))
if len(df):
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    print("entry min:", df["entry_time"].min())
    print("entry max:", df["entry_time"].max())

