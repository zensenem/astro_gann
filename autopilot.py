#!/usr/bin/env python3
# autopilot.py — optimize -> report -> predict (ccxt robust pagination + fallback)

import argparse, json, sys, time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from astro_gann_indicator import AstroGannIndicator
from indicator_upgrade_kit import build_feature_frame, random_search_opt, backtest_report, generate_signals

def need_reopt(params_path: Path, reopt_days: int) -> bool:
    if not params_path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(params_path.stat().st_mtime)
    return age > timedelta(days=reopt_days)

def ensure_outdir(base_out: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = base_out / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def timeframe_to_ms(tf: str) -> int:
    tf = tf.strip()
    num = int(''.join([c for c in tf if c.isdigit()]))
    unit = ''.join([c for c in tf if c.isalpha()])
    return {"m":60000,"h":3600000,"d":86400000,"w":604800000,"M":2592000000}[unit] * num

def fetch_ccxt(exchange_name: str, symbol: str, timeframe: str, days: int, out_csv: Path) -> bool:
    try:
        import ccxt
    except Exception as e:
        print("[HATA] ccxt modülü yok (pip install ccxt):", e)
        return False

    # Borsayı güvenli parametrelerle aç
    try:
        ex_class = getattr(ccxt, exchange_name)
    except AttributeError:
        print(f"[UYARI] '{exchange_name}' bulunamadı, 'binance' kullanılacak.")
        ex_class = ccxt.binance
    ex = ex_class({
        "enableRateLimit": True,
        "timeout": 30000,  # 30 sn
        "options": {"defaultType": "spot"},
    })

    # Pazarları yükle (exponential backoff)
    ok = False
    for i in range(8):
        try:
            ex.load_markets(reload=True)
            ok = True
            break
        except Exception as e:
            wait = 0.5 * (2 ** i)
            print(f"[UYARI] load_markets hatası: {e.__class__.__name__} -> {wait:.1f}s bekle")
            time.sleep(wait)
    if not ok:
        print("[HATA] Pazar bilgisi alınamadı.")
        return False

    # Sembol eşlemesi (gerekiyorsa alternatif dene)
    m_symbol = symbol
    if m_symbol not in ex.markets and symbol.replace("-", "/") in ex.markets:
        m_symbol = symbol.replace("-", "/")
    if m_symbol not in ex.markets and "/USD" in symbol and (symbol.replace("/USD","/USDT")) in ex.markets:
        m_symbol = symbol.replace("/USD","/USDT")
    if m_symbol not in ex.markets and "/USDT" in symbol and (symbol.replace("/USDT","/USD")) in ex.markets:
        m_symbol = symbol.replace("/USDT","/USD")
    if m_symbol not in ex.markets:
        # Son çare: BTC ile başlayan bir sembol bul
        cands = [s for s in ex.symbols if s.startswith("BTC/")]
        print(f"[UYARI] {symbol} bulunamadı, adaylar: {cands[:5]}")
        if cands:
            m_symbol = cands[0]
        else:
            print("[HATA] Uygun sembol bulunamadı.")
            return False

    tf_ms = timeframe_to_ms(timeframe)
    ms_now = ex.milliseconds()
    since = ms_now - days * 86400000
    hard_limit = 1000
    rows, loops = [], 0
    print(f"[Bilgi] ccxt {ex.id} {m_symbol} {timeframe} ~{days}g çekiliyor (sayfalama)...")

    last_seen = None
    while since < ms_now and loops < 20000:
        try:
            ohlcv = ex.fetch_ohlcv(m_symbol, timeframe=timeframe, since=since, limit=hard_limit)
        except Exception as e:
            wait = min(10.0, 0.5 * (2 ** min(loops, 6)))
            print(f"[UYARI] fetch_ohlcv: {e.__class__.__name__} -> {wait:.1f}s bekle")
            time.sleep(wait)
            continue
        if not ohlcv:
            break
        last_ts = ohlcv[-1][0]
        # İlerleme yoksa kır
        if last_seen is not None and last_ts <= last_seen:
            break
        rows += ohlcv
        last_seen = last_ts
        since = last_ts + tf_ms
        loops += 1
        time.sleep(0.2)

    if not rows:
        print("[UYARI] ccxt boş döndü.")
        return False

    df = pd.DataFrame(rows, columns=["Date","Open","High","Low","Close","Volume"])
    df["Date"] = pd.to_datetime(df["Date"], unit="ms", utc=True)
    # Çakışma/tekrarları temizle
    df = df.drop_duplicates(subset=["Date"]).dropna(subset=["Open","High","Low","Close"]).sort_values("Date")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(f"[Bilgi] ccxt verisi yazıldı -> {out_csv} (satır: {len(df)})")
    if len(df) < days * 20:
        print("[UYARI] Satır sayısı beklenenden düşük; borsa limiti/ağ olabilir.")
    return True

def maybe_fetch_market_data(args) -> Path:
    csv_path = Path(args.csv)
    src = args.data_source
    if src == "csv":
        return csv_path
    if src == "yfinance":
        try:
            import yfinance as yf
            t = yf.Ticker(args.yfinance)
            df = t.history(interval=args.interval, period=args.period, auto_adjust=False)[["Open","High","Low","Close","Volume"]].dropna()
            if df.index.tz is None: df.index = df.index.tz_localize("UTC")
            else: df.index = df.index.tz_convert("UTC")
            df.index.name = "Date"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, date_format="%Y-%m-%dT%H:%M:%SZ")
            print(f"[Bilgi] yfinance verisi yazıldı -> {csv_path}")
            return csv_path
        except Exception as e:
            print("[UYARI] yfinance başarısız:", e.__class__.__name__, "-", e, "-> ccxt'ye düşülüyor")
            # ccxt’ye düş
            if fetch_ccxt(args.ccxt_exchange, args.ccxt_symbol, args.ccxt_timeframe, args.ccxt_days, csv_path):
                return csv_path
            print("[HATA] Veri alınamadı.")
            sys.exit(2)
    if src == "ccxt":
        if fetch_ccxt(args.ccxt_exchange, args.ccxt_symbol, args.ccxt_timeframe, args.ccxt_days, csv_path):
            return csv_path
        print("[HATA] ccxt veri çekemedi.")
        sys.exit(2)
    return csv_path

def main():
    ap = argparse.ArgumentParser(description="Astro/Gann autopilot — optimize->report->predict")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--predict", type=int, default=24)
    ap.add_argument("--fee_bps", type=float, default=5.0)
    ap.add_argument("--params_path", default="best_params.json")
    ap.add_argument("--reopt_days", type=int, default=14)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--force_reopt", action="store_true")

    ap.add_argument("--data_source", choices=["csv","yfinance","ccxt"], default="csv")
    ap.add_argument("--yfinance", default=None)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--period", default="730d")

    ap.add_argument("--ccxt_exchange", default="binance")
    ap.add_argument("--ccxt_symbol", default="BTC/USDT")
    ap.add_argument("--ccxt_timeframe", default="1h")
    ap.add_argument("--ccxt_days", type=int, default=730)

    args = ap.parse_args()

    # 0) Veri
    csv_path = maybe_fetch_market_data(args)

    # 1) İndikatör + feature
    ind = AstroGannIndicator()
    if not ind.load_data(file_path=str(csv_path), source='csv'):
        print("[HATA] CSV yüklenemedi:", csv_path)
        sys.exit(2)
    feat = build_feature_frame(ind)

    # 2) Parametreler
    params_path = Path(args.params_path)
    if args.force_reopt or need_reopt(params_path, args.reopt_days):
        print(f"[Bilgi] Yeniden optimizasyon (iters={args.iters})...")
        best = random_search_opt(feat, horizon=args.horizon, iters=args.iters, fee_bps=args.fee_bps)
        params_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Bilgi] En iyi parametreler kaydedildi -> {params_path}")
        params = best
    else:
        print(f"[Bilgi] Mevcut parametreler güncel -> {params_path}")
        params = json.loads(params_path.read_text(encoding="utf-8"))

    # 3) Çıktı klasörü
    run_dir = ensure_outdir(Path(args.out_dir))

    # 4) Rapor
    rep = backtest_report(feat, params, horizon=args.horizon, fee_bps=args.fee_bps)
    rep_path = run_dir / "backtest_report.csv"
    rep.to_csv(rep_path)
    print("[Bilgi] Rapor yazıldı ->", rep_path)

    # 5) Sinyaller
    sig = generate_signals(feat, params, horizon=args.horizon).iloc[-args.predict:]
    sig_path = run_dir / "signals_last.csv"
    sig.to_csv(sig_path)
    print("[Bilgi] Sinyaller yazıldı ->", sig_path)

    # 6) Özet
    summary = {
        "csv": str(csv_path),
        "params_path": str(params_path),
        "run_dir": str(run_dir),
        "horizon": args.horizon,
        "iters": args.iters,
        "predict": args.predict,
        "fee_bps": args.fee_bps,
        "rows": len(feat)
    }
    (run_dir/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Bitti] Özet ->", run_dir/"summary.json")

if __name__ == "__main__":
    main()
