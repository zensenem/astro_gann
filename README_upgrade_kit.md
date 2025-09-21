
# Astro/Gann Indicator — Consistency-First Upgrade Kit (Vertex‑Ready, Local‑First)

Bu kit, mevcut `astro_gann_indicator.py` sınıfını **bozmadan** üstüne bir katman ekler:

- **Yerel-İlk**: CSV veya kendi veri kaynağınla çalışır; hiçbir bulut servisine mecbur değilsin.
- **Tutarlılık Odaklı**: Ağırlık ve eşiklerini *zaman serisi* walk‑forward test ile kalibre eder.
- **Vertex‑Hazır**: Fonksiyonlar modüler; istersen aynı adımları Vertex AI Pipeline bileşenleri olarak kolayca paketleyebilirsin.
- **Minimum maliyet**: BigQuery/Endpoints zorunluluğu yok. İsteyen için BigQuery okuma stub’u da var (opsiyonel).

## Dosyalar

- `indicator_upgrade_kit.py` — veri kümesi üretimi, walk‑forward backtest, rastgele arama ile ağırlık/threshold kalibrasyonu, basit sinyal üretimi ve CLI.
- `vertex_pipeline_stub.py` — (opsiyonel) aynı adımların Vertex AI bileşen olarak paketlenmiş iskeleti.
- `config_example.json` — örnek yapılandırma (hedef ufku, ücret/komisyon, split sayıları vs.).

## Hızlı Başlangıç

> Önkoşul: `astro_gann_indicator.py` dosyan bu klasörde olmalı ve CSV’nin sütunları `Date,Open,High,Low,Close,Volume` şeklinde olmalı.

```bash
# 1) Ağırlık & eşik kalibrasyonu (rastgele arama)
python indicator_upgrade_kit.py --csv your_data.csv --optimize --iters 300 --horizon 1

# 2) En iyi ayarlarla backtest raporu
python indicator_upgrade_kit.py --csv your_data.csv --report --horizon 1

# 3) Son gün(ler) için sinyal üret
python indicator_upgrade_kit.py --csv your_data.csv --predict 10
```

> Not: `--horizon` = ileri bakış adımı (örn. 1 = bir sonraki bar). Dakikalık / saatlik serilerde bar karşılığıdır.

## Vertex’e Geçmek İstersen

- `vertex_pipeline_stub.py` içindeki fonksiyonlar zaten @component yapısına uygun yazıldı.
- BigQuery veya GCS kullanacaksan yalnızca I/O tarafını doldurman yeterli.
- Endpoints yayımlamaya gerek yok; batch prediction çıktısını yerelde kullanabilirsin.

Keyifli kalibrasyonlar ✨
