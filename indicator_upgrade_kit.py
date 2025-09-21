# -*- coding: utf-8 -*-
"""
Indicator Upgrade Kit (sağlam sürüm)
- build_feature_frame(ind)
- random_search_opt(feat, horizon, iters, fee_bps)
- backtest_report(feat, params, horizon, fee_bps)  [cooldown/min-hold entegre]
- generate_signals(feat, params, horizon)          [cooldown/min-hold entegre]
"""

import os, math, json, random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    import talib as ta
except Exception:
    ta = None  # TA-Lib yoksa teknikler sınırlı çalışır

# --- Cooldown / Min-Hold ayarları (ortamdan oynatılabilir) ---
COOLDOWN_BARS = int(os.getenv("COOLDOWN_BARS", "3"))
MIN_HOLD_BARS = int(os.getenv("MIN_HOLD_BARS", "3"))


# ----------------------------- Yardımcılar -----------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Date index'i garanti eder, UTC'ye çevirir ve sıralar."""
    x = df.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        if "Date" in x.columns:
            x["Date"] = pd.to_datetime(x["Date"], errors="coerce", utc=True)
            x = x.set_index("Date")
        else:
            x.index = pd.to_datetime(x.index, errors="coerce", utc=True)
    x = x.sort_index()
    return x


def _coerce_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Open", "High", "Low", "Close"])
    return x


def _calc_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Temel teknik göstergeler ve kaba bir Tech_Score üretir."""
    x = df.copy()
    if ta is None:
        # TA-Lib yoksa minimal Tech_Score: hareketli ortalama momentumu
        close = x["Close"]
        ma_fast = close.rolling(12).mean()
        ma_slow = close.rolling(26).mean()
        tech = (ma_fast - ma_slow) / (x["Close"].rolling(50).std(ddof=0) + 1e-9)
        x["Tech_Score"] = tech.fillna(0.0)
        return x

    c = x["Close"].values
    h = x["High"].values
    l = x["Low"].values

    rsi = ta.RSI(c, timeperiod=14)
    macd, macds, macdh = ta.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    slowk, slowd = ta.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowk_matype=0,
                            slowd_period=3, slowd_matype=0)
    upper, middle, lower = ta.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    atr = ta.ATR(h, l, c, timeperiod=14)

    x["RSI"] = rsi
    x["MACD"] = macd
    x["MACD_Signal"] = macds
    x["MACD_Hist"] = macdh
    x["Stoch_K"] = slowk
    x["Stoch_D"] = slowd
    x["BB_Upper"] = upper
    x["BB_Middle"] = middle
    x["BB_Lower"] = lower
    x["ATR"] = atr

    # Basit normalize edilmiş Tech_Score (0 merkezli ~[-1,1] civarı)
    rsi_s   = 1.0 - (pd.Series(rsi, index=x.index) / 50.0)           # RSI>50 -> negatif (mean-revert)
    macd_s  = pd.Series(macdh, index=x.index) / (x["Close"].rolling(50).std(ddof=0) + 1e-9)
    stoch_s = 1.0 - (pd.Series(slowk, index=x.index) / 50.0)
    bb_pos  = (x["Close"] - pd.Series(lower, index=x.index)) / (pd.Series(upper, index=x.index) - pd.Series(lower, index=x.index) + 1e-9)
    bb_s    = 0.5 - bb_pos
    tech    = (0.30*rsi_s + 0.30*macd_s + 0.20*stoch_s + 0.20*bb_s).astype(float)

    x["Tech_Score"] = pd.to_numeric(tech, errors="coerce").fillna(0.0)
    return x


# ----------------------- Dışa açık API fonksiyonları -----------------------
def build_feature_frame(ind) -> pd.DataFrame:
    """
    AstroGannIndicator örneğinden feature frame üretir.
    - Teknikler hesaplanır.
    - Astro_Score / Gann_Score yoksa 0.0 eklenir (fallback).
    """
    if not hasattr(ind, "data"):
        raise ValueError("Indicator 'ind' veri içermiyor.")

    df = _ensure_datetime_index(ind.data)
    df = _coerce_ohlcv(df)
    df = _calc_ta(df)

    # Astro/Gann skorları yoksa sıfırla (fallback)
    for col in ["Astro_Score", "Gann_Score"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def time_series_folds(n: int, n_splits: int = 5, min_train: int = 200, gap: int = 0):
    """Walk-forward için basit katmanlayıcı."""
    if n < min_train + 5:
        return
    step = max(1, (n - min_train) // n_splits)
    for i in range(n_splits):
        end = min_train + i*step
        tr_end = max(0, end - gap)
        va_end = min(end + step, n)
        tr = list(range(0, tr_end))
        va = list(range(tr_end, va_end))
        if len(va) >= 5:
            yield (tr, va)


def apply_cooldown_signals(df: pd.DataFrame, up_th: float, down_th: float,
                           min_hold: int = 3, cooldown: int = 3) -> pd.DataFrame:
    """
    'Total' -> ham sinyal; min-hold & cooldown ile düzleştirip 'Signal' üretir.
    """
    x = df.copy()
    total = pd.to_numeric(x["Total"], errors="coerce").fillna(0.0).values
    out = []
    pos, hold, cd = 0, 0, 0
    up_th  = float(up_th)
    down_th = float(down_th)

    for val in total:
        raw = 1 if val >= up_th else (-1 if val <= down_th else 0)
        if pos == 0:
            if cd > 0:
                cd -= 1
                sig = 0
            else:
                if raw != 0:
                    pos = raw
                    hold = int(min_hold)
                    sig = pos
                else:
                    sig = 0
        else:
            if hold > 0:
                hold -= 1
                sig = pos
            else:
                if raw * pos < 0:  # karşı sinyal geldi -> flip
                    pos = raw
                    hold = int(min_hold)
                    cd = int(cooldown)
                    sig = pos
                else:
                    sig = pos
        out.append(sig)

    x["Signal"] = out
    x["Direction"] = x["Signal"].map({1: "Yükseliş", -1: "Düşüş", 0: "Nötr"})
    return x


def _bt_metrics_from_signals(df: pd.DataFrame, pos: pd.Series, fee_bps: float = 5.0, horizon: int = 1) -> Dict[str, float]:
    """Basit getiri/Sharpe/Sortino/hit-rate + trade sayısı."""
    close = pd.to_numeric(df["Close"], errors="coerce")
    fwd = (close.shift(-horizon) / close - 1.0)
    r = (pos.astype(float) * fwd).fillna(0.0)

    ch = pos.fillna(0).astype(float).diff().abs().fillna(0.0)
    fees = ch * (fee_bps / 10000.0)
    r = (r - fees).dropna()

    if len(r) == 0:
        return dict(sharpe=0.0, sortino=0.0, hit_rate=0.0, avg_return=0.0, trades=int((ch > 0).sum()))

    mean = float(r.mean())
    std  = float(r.std(ddof=0))
    neg  = r[r < 0]
    dd   = float(neg.std(ddof=0)) if len(neg) > 0 else 0.0
    sharpe  = (mean / std) if std > 0 else 0.0
    sortino = (mean / dd)  if dd  > 0 else 0.0
    hit     = float((r > 0).mean())
    trades  = int((ch > 0).sum())

    return dict(sharpe=sharpe, sortino=sortino, hit_rate=hit, avg_return=mean, trades=trades)


def backtest_report(feat: pd.DataFrame, params: Dict[str, float], horizon: int = 1, fee_bps: float = 5.0) -> pd.DataFrame:
    """Cooldown'lı walk-forward rapor."""
    f = feat.copy()
    for col in ["Astro_Score", "Gann_Score", "Tech_Score"]:
        if col not in f.columns:
            f[col] = 0.0
        f[col] = pd.to_numeric(f[col], errors="coerce").fillna(0.0)

    w_astro = float(params.get("w_astro", 0.33))
    w_gann  = float(params.get("w_gann",  0.33))
    w_tech  = float(params.get("w_tech",  0.34))
    f["Total"] = (w_astro * f["Astro_Score"] + w_gann * f["Gann_Score"] + w_tech * f["Tech_Score"]).astype(float)

    window = max(50, max(5, len(f)//10))
    step   = max(5, window//4)

    rows = []
    for start in range(0, max(1, len(f) - window + 1), step):
        seg = f.iloc[start:start+window]
        if len(seg) < 5:
            continue
        cooled = apply_cooldown_signals(
            seg[["Total"]].join(seg[["Close"]]),
            params.get("up_th", 0.1),
            params.get("down_th", -0.1),
            min_hold=MIN_HOLD_BARS,
            cooldown=COOLDOWN_BARS
        )
        met = _bt_metrics_from_signals(seg, cooled["Signal"], fee_bps=fee_bps, horizon=horizon)
        met["start"] = seg.index[0]
        met["end"]   = seg.index[-1]
        rows.append(met)

    return pd.DataFrame(rows)


def random_search_opt(feat: pd.DataFrame, horizon: int = 1, iters: int = 300, fee_bps: float = 5.0) -> Dict[str, float]:
    """Dirichlet ağırlık + eşik araması; hedef = pencerelerde ortalama Sharpe."""
    f = feat.copy()
    n = len(f)
    if n < 300:
        raise ValueError("Walk-forward için yeterli veri yok.")

    dyn_splits = 5
    dyn_min_train = max(200, n // 5)
    folds = list(time_series_folds(n, n_splits=dyn_splits, min_train=dyn_min_train, gap=0))
    if not folds:
        raise ValueError("Walk-forward için yeterli veri yok.")

    rng = np.random.default_rng()
    best, best_obj = None, -1e9

    for _ in range(iters):
        w = rng.dirichlet([1.0, 1.0, 1.0])
        params = dict(
            w_astro=float(w[0]),
            w_gann =float(w[1]),
            w_tech =float(w[2]),
            up_th  =float(rng.uniform(0.05, 0.2)),
            down_th=float(-rng.uniform(0.05, 0.2)),
        )

        scores = []
        for tr, va in folds:
            seg = f.iloc[0:va[-1]+1]  # basit ileri kayan segment
            rep = backtest_report(seg, params, horizon=horizon, fee_bps=fee_bps)
            if not rep.empty:
                scores.append(float(rep["sharpe"].mean()))

        obj = float(sum(scores)/len(scores)) if scores else -1e9
        if obj > best_obj:
            best_obj = obj
            best = dict(params)
            best["objective"] = obj

    return best


def generate_signals(feat: pd.DataFrame, params: Dict[str, float], horizon: int = 1) -> pd.DataFrame:
    """Toplam skordan sinyal üret, cooldown/min-hold uygula."""
    f = feat.copy()
    for col in ["Astro_Score", "Gann_Score", "Tech_Score"]:
        if col not in f.columns:
            f[col] = 0.0
        f[col] = pd.to_numeric(f[col], errors="coerce").fillna(0.0)

    total = (
        float(params.get("w_astro", 0.33)) * f["Astro_Score"]
        + float(params.get("w_gann", 0.33)) * f["Gann_Score"]
        + float(params.get("w_tech", 0.34)) * f["Tech_Score"]
    ).astype(float)

    out = pd.DataFrame(index=f.index)
    out["Total"] = pd.to_numeric(total, errors="coerce").fillna(0.0)
    out = out.join(f[["Close"]], how="left")

    out = apply_cooldown_signals(
        out,
        params.get("up_th", 0.1),
        params.get("down_th", -0.1),
        min_hold=MIN_HOLD_BARS,
        cooldown=COOLDOWN_BARS
    )
    return out
