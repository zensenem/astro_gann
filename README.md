# Astro Gann

Yerel veri ile çalışır. Örnek kullanım:

```bash
conda activate astroind
export MPLBACKEND=Agg
cd ~/Desktop/astro_gann

MIN_HOLD_BARS=8 COOLDOWN_BARS=3 \
python autopilot.py --data_source csv --csv data/BTCUSDT_2h.csv \
  --horizon 1 --iters 1 --predict 24 --fee_bps 10.0 \
  --params_path configs/BTCUSDT_2h_best_quantile.json \
  --reopt_days 9999 --out_dir runs/BTCUSDT_2h

python report_latest.py && open report_latest.html
Not: data/*.csv ve runs/ .gitignore ile VCS dışında.
