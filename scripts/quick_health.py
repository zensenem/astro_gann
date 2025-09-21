from pathlib import Path
import pandas as pd, json

def latest_with(file, root):
    runs = [p for p in root.glob("*") if (p/file).exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None

for base in ("runs/BTCUSDT_2h","runs/BTCUSDT_1h"):
    r = latest_with("signals_last.csv", Path(base))
    if not r:
        print("No runs in", base)
        continue
    sig = pd.read_csv(r/"signals_last.csv", parse_dates=["Date"])
    s   = json.loads((r/"summary.json").read_text(encoding="utf-8"))
    params = json.loads(Path(s["params_path"]).read_text(encoding="utf-8"))
    up, down = float(params["up_th"]), float(params["down_th"])
    total = sig["Total"].astype(float)
    flips = int(sig["Signal"].astype(float).diff().fillna(0).ne(0).sum())
    print(f"\n[{Path(base).name} -> {r.name}] rows={len(total)} up={up:.5f} down={down:.5f} flips={flips}")
    print("Total min/med/max:", float(total.min()), float(total.median()), float(total.max()))
    print("Crosses  >up:", int((total>up).sum()), " <down:", int((total<down).sum()))
    print("Signals:", sig["Signal"].value_counts().to_dict())
