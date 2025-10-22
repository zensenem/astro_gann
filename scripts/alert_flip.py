from pathlib import Path
import pandas as pd, subprocess, json, time

ROOT = Path.home()/ "Desktop" / "astro_gann"
STATE = ROOT / "state" / "last_flip.txt"

def latest_run(base: Path) -> Path|None:
    runs = [p for p in base.glob("*") if (p/"signals_last.csv").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None

def notify(title: str, msg: str):
    try:
        subprocess.run(["osascript","-e",f'display notification "{msg}" with title "{title}"'],
                       check=False)
        # İstersen ses de çal:  subprocess.run(["afplay","/System/Library/Sounds/Glass.aiff"])
    except Exception as e:
        print("Notify error:", e)

def main():
    base = ROOT / "runs" / "BTCUSDT_2h"
    r = latest_run(base)
    if not r:
        print("Koşu yok."); return
    df = pd.read_csv(r/"signals_last.csv", parse_dates=["Date"])
    sig = df["Signal"].astype(int)
    if len(sig) < 2: 
        print("Yeterli bar yok."); return
    last, prev = sig.iloc[-1], sig.iloc[-2]
    if last == prev:
        print("Flip yok."); return

    mode = "LONG" if last==1 else "SHORT"
    when = df["Date"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
    alert_id = f"{r.name}:{int(df.index[-1])}:{mode}"

    STATE.parent.mkdir(parents=True, exist_ok=True)
    before = STATE.read_text().strip() if STATE.exists() else ""
    if before == alert_id:
        print("Bu flip için uyarı zaten gönderilmiş.")
        return

    # Eşikleri de bilgi olarak ekle
    params_path = json.loads((r/"summary.json").read_text(encoding="utf-8"))["params_path"]
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    up, down = float(params["up_th"]), float(params["down_th"])

    msg = f"{mode} @ {when}  | up={up:.3f} down={down:.3f}"
    print("ALERT:", msg)
    notify("AstroGann Flip", msg)
    STATE.write_text(alert_id)

if __name__ == "__main__":
    main()
