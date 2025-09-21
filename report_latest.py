#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, json, sys, traceback, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def latest_run(base):
    base = Path(base)
    runs = [p for p in base.glob("*") if (p/"backtest_report.csv").exists()]
    if not runs: return None
    return max(runs, key=lambda p: p.stat().st_mtime)

def resolve_csv_path(run_dir: Path, csv_field: str) -> Path:
    """
    summary.json içindeki 'csv' alanını proje köküne göre doğru çözer.
    run_dir = .../astro_gann/runs/<tf>/<timestamp>
    proje kökü = run_dir.parents[2]  (yani .../astro_gann)
    """
    p = Path(csv_field)
    if p.is_absolute():
        return p
    proj_root = run_dir.parents[2] if len(run_dir.parents) >= 3 else run_dir.resolve()
    cand = (proj_root / p)           # genelde data/.. burada
    if cand.exists():
        return cand
    alt = (proj_root / p.name)       # son çare: sadece dosya adıyla dene
    if alt.exists():
        return alt
    # Eski hatalı mantığa düşersek yine de bir yol döndür (varsa okuyacağız)
    return cand

def load_metrics(run_dir):
    rep = pd.read_csv(run_dir/"backtest_report.csv")
    windows = len(rep)
    sharpe_med = float(rep["sharpe"].median())
    sortino_med = float(rep["sortino"].median())
    pos_sharpe = float((rep["sharpe"]>0).mean())*100.0
    pos_sortino= float((rep["sortino"]>0).mean())*100.0
    trades_med = int(rep["trades"].median())

    sig = pd.read_csv(run_dir/"signals_last.csv", parse_dates=["Date"])
    sig["Signal"] = sig["Signal"].astype(float)
    flips = int((sig["Signal"].diff().fillna(0)!=0).sum())
    sig_counts = sig["Signal"].value_counts(dropna=False).to_dict()

    s = json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))
    params = json.loads(Path(s["params_path"]).read_text(encoding="utf-8"))
    up, down = float(params.get("up_th",0)), float(params.get("down_th",0))

    # CSV yolunu doğru çöz
    csv_path = resolve_csv_path(run_dir, s.get("csv",""))
    preview = run_dir/"preview.png"
    preview_exists = preview.exists()

    # Önizleme yoksa üret, ama CSV okunamazsa raporu ÇÖKERTME
    if not preview_exists:
        try:
            px  = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")["Close"]
            ss  = sig.set_index("Date")
            df = pd.concat([px, ss["Total"], ss["Signal"]], axis=1).dropna().tail(300)
            df.columns=["Close","Total","Signal"]
            fig = plt.figure(figsize=(12,6)); ax=plt.gca()
            df["Close"].plot(ax=ax)
            ax2 = ax.twinx(); df["Total"].plot(ax=ax2, alpha=0.45)
            ax2.axhline(up,   linestyle="--", alpha=0.4)
            ax2.axhline(down, linestyle="--", alpha=0.4)
            ax.set_title(f"{run_dir.parent.name} Close & Total (last 300) | {run_dir.name}")
            ax.set_ylabel("Close"); ax2.set_ylabel("Total")
            plt.tight_layout(); plt.savefig(preview, dpi=160); plt.close(fig)
            preview_exists = True
        except Exception as e:
            print("Uyarı: fiyat CSV okunamadı veya grafik üretilemedi:", csv_path)
            traceback.print_exc()

    return {
        "windows": windows,
        "sharpe_med": sharpe_med,
        "sortino_med": sortino_med,
        "pos_sharpe": pos_sharpe,
        "pos_sortino": pos_sortino,
        "trades_med": trades_med,
        "flips": flips,
        "sig_counts": {int(k): int(v) for k,v in sig_counts.items() if pd.notna(k)},
        "up": up, "down": down,
        "preview": str(preview) if preview_exists else "",
        "run_name": run_dir.name,
        "csv": str(csv_path)
    }

def html_block(tf, m):
    sig_counts = " | ".join([f"{int(k)}: {v}" for k,v in sorted(m["sig_counts"].items())]) or "—"
    img_html = f'<img src="{m["preview"]}" alt="preview" style="max-width:100%;border:1px solid #ccc;border-radius:6px;">' if m["preview"] else "<i>Önizleme üretilemedi (CSV okunamadı).</i>"
    return f"""
    <section>
      <h2>{tf} — {m["run_name"]}</h2>
      <ul>
        <li><b>Windows:</b> {m["windows"]}</li>
        <li><b>Sharpe medyan:</b> {m["sharpe_med"]:.4f}  (>%0: {m["pos_sharpe"]:.1f}%)</li>
        <li><b>Sortino medyan:</b> {m["sortino_med"]:.4f}  (>%0: {m["pos_sortino"]:.1f}%)</li>
        <li><b>Trades medyan:</b> {m["trades_med"]}</li>
        <li><b>Eşikler:</b> up_th={m["up"]:.6f}, down_th={m["down"]:.6f}</li>
        <li><b>Flip sayısı:</b> {m["flips"]} | <b>Signal dağılımı:</b> {sig_counts}</li>
        <li><b>CSV:</b> <code>{m["csv"]}</code></li>
      </ul>
      {img_html}
    </section>
    """

def main():
    proj = Path(__file__).resolve().parent
    blocks=[]
    for base in ["runs/BTCUSDT_1h","runs/BTCUSDT_2h"]:
        r = latest_run(proj/base)
        if r:
            try:
                m = load_metrics(r)
                tf = "1h" if "1h" in base else "2h"
                blocks.append(html_block(tf, m))
            except Exception:
                traceback.print_exc()
    html = f"""
    <html><head><meta charset="utf-8"><title>AstroGann Quick Report</title>
    <style>
    body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif; max-width:980px;margin:32px auto;padding:0 16px;}}
    h1{{margin-top:0}}
    section{{margin:24px 0 40px}}
    ul{{line-height:1.6}}
    code{{background:#f6f8fa;padding:2px 6px;border-radius:4px}}
    .note{{font-size:13px;color:#666;margin-top:40px}}
    </style></head><body>
    <h1>AstroGann Quick Report</h1>
    {''.join(blocks) if blocks else '<p>Koşu bulunamadı.</p>'}
    <p class="note">Bu bir yatırım tavsiyesi değildir.</p>
    </body></html>
    """
    out = proj/"report_latest.html"
    out.write_text(html, encoding="utf-8")
    print("Wrote:", out)

if __name__ == "__main__":
    main()
