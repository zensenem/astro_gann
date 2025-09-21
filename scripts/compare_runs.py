from pathlib import Path
import pandas as pd, json

rows=[]
for r in sorted([p for p in Path("runs/BTCUSDT_2h").glob("*") if (p/"backtest_report.csv").exists()],
                key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
    s   = json.loads((r/"summary.json").read_text(encoding="utf-8"))
    rep = pd.read_csv(r/"backtest_report.csv")
    rows.append({
        "run": r.name,
        "params": Path(s["params_path"]).name,
        "sharpe_med": float(rep["sharpe"].median()),
        "sortino_med": float(rep["sortino"].median()),
        "trades_med":  int(rep["trades"].median())
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
