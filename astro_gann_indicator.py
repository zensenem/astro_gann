import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pytz
import ephem
from skyfield.api import load, wgs84
from skyfield.data import mpc
import talib
import warnings
warnings.filterwarnings('ignore')
import os

class AstroGannIndicator:
    """
    AstroGann İndikatörü - Astroloji ve Gann teorisini birleştiren gelişmiş borsa analiz aracı
    
    Bu indikatör, astrolojik faktörleri, Gann teorisini ve teknik analiz göstergelerini
    birleştirerek piyasa yönü tahminleri sunar.
    """
    
    def __init__(self):
        self.version = "1.0.0"
        self.data = None
        self.predictions = None
        self.current_date = datetime.now()
        self.planets = {
            'Sun': ephem.Sun(),
            'Moon': ephem.Moon(),
            'Mercury': ephem.Mercury(),
            'Venus': ephem.Venus(),
            'Mars': ephem.Mars(),
            'Jupiter': ephem.Jupiter(),
            'Saturn': ephem.Saturn(),
            'Uranus': ephem.Uranus(),
            'Neptune': ephem.Neptune(),
            'Pluto': ephem.Pluto()
        }
        
        # Gann açıları
        self.gann_angles = [
            (1, 1),  # 45 derece
            (1, 2),  # 26.25 derece
            (1, 3),  # 18.75 derece
            (1, 4),  # 15 derece
            (1, 8),  # 7.5 derece
            (2, 1),  # 63.75 derece
            (3, 1),  # 71.25 derece
            (4, 1),  # 75 derece
            (8, 1)   # 82.5 derece
        ]
        
        # Fibonacci seviyeleri
        self.fibonacci_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
        
        # Gann döngüleri (gün cinsinden)
        self.gann_cycles = [30, 45, 60, 90, 120, 180, 270, 360]
        
        # Astrolojik açılar (derece cinsinden)
        self.aspects = {
            'Conjunction': 0,      # Kavuşum
            'Opposition': 180,     # Karşıt
            'Trine': 120,          # Üçgen
            'Square': 90,          # Kare
            'Sextile': 60,         # Altıgen
            'Quincunx': 150,       # Yüzellilik
            'Semi-sextile': 30,    # Yarı-altıgen
            'Semi-square': 45,     # Yarı-kare
            'Sesquiquadrate': 135  # Bir-buçuk-kare
        }
        
        # Gezegensel etkiler (piyasa üzerindeki etki puanları)
        self.planet_influences = {
            'Sun': 0.8,       # Güneş - Genel trend
            'Moon': 0.9,      # Ay - Kısa vadeli dalgalanmalar
            'Mercury': 0.7,   # Merkür - İletişim, ticaret
            'Venus': 0.6,     # Venüs - Değer, para
            'Mars': 0.75,     # Mars - Enerji, ani hareketler
            'Jupiter': 0.85,  # Jüpiter - Genişleme, büyüme
            'Saturn': 0.8,    # Satürn - Sınırlama, disiplin
            'Uranus': 0.65,   # Uranüs - Ani değişimler
            'Neptune': 0.5,   # Neptün - Yanılsama, belirsizlik
            'Pluto': 0.7      # Plüton - Derin dönüşüm
        }
        
        # Teknik gösterge ağırlıkları
        self.indicator_weights = {
            'RSI': 0.08,
            'MACD': 0.08,
            'Stochastic': 0.07,
            'Bollinger': 0.07,
            'ATR': 0.05,
            'OBV': 0.06,
            'ADX': 0.06,
            'EMA': 0.08,
            'Ichimoku': 0.07,
            'Fibonacci': 0.08,
            'Gann': 0.15,
            'Astro': 0.15
        }
        
    def load_data(self, file_path=None, symbol=None, start_date=None, end_date=None, source='csv'):
        """
        Veri yükleme fonksiyonu
        
        Parameters:
        -----------
        file_path : str, optional
            CSV dosya yolu
        symbol : str, optional
            Sembol adı (API kullanımı için)
        start_date : str, optional
            Başlangıç tarihi (YYYY-MM-DD formatında)
        end_date : str, optional
            Bitiş tarihi (YYYY-MM-DD formatında)
        source : str, optional
            Veri kaynağı ('csv', 'api')
        """
        if source == 'csv' and file_path:
            try:
                self.data = pd.read_csv(file_path)
                
                # Tarih sütununu datetime formatına dönüştür
                if 'Date' in self.data.columns:
                    self.data['Date'] = pd.to_datetime(self.data['Date'])
                    self.data.set_index('Date', inplace=True)
                
                # Gerekli sütunların varlığını kontrol et
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in self.data.columns:
                        raise ValueError(f"Veri setinde {col} sütunu bulunamadı")
                
                print(f"Veri başarıyla yüklendi. Toplam {len(self.data)} kayıt.")
                return True
                
            except Exception as e:
                print(f"Veri yükleme hatası: {str(e)}")
                return False
        
        elif source == 'api' and symbol:
            # API entegrasyonu burada yapılacak
            # Örnek olarak, yalancı veri oluşturalım
            try:
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                if not end_date:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                # Örnek veri oluştur
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                self.data = pd.DataFrame({
                    'Open': np.random.normal(100, 5, len(date_range)),
                    'High': np.random.normal(105, 5, len(date_range)),
                    'Low': np.random.normal(95, 5, len(date_range)),
                    'Close': np.random.normal(102, 5, len(date_range)),
                    'Volume': np.random.normal(1000000, 200000, len(date_range))
                }, index=date_range)
                
                print(f"API'den veri başarıyla yüklendi. Toplam {len(self.data)} kayıt.")
                return True
                
            except Exception as e:
                print(f"API veri yükleme hatası: {str(e)}")
                return False
        
        else:
            print("Geçersiz veri kaynağı veya eksik parametreler")
            return False
    
    def calculate_technical_indicators(self):
        """
        Teknik göstergeleri hesaplar
        """
        if self.data is None:
            print("Önce veri yüklemelisiniz")
            return False
        
        try:
            # RSI (Göreceli Güç Endeksi)
            self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)
            
            # MACD (Hareketli Ortalama Yakınsama/Iraksama)
            macd, macd_signal, macd_hist = talib.MACD(
                self.data['Close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            self.data['MACD'] = macd
            self.data['MACD_Signal'] = macd_signal
            self.data['MACD_Hist'] = macd_hist
            
            # Stokastik Osilatör
            slowk, slowd = talib.STOCH(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'], 
                fastk_period=5, 
                slowk_period=3, 
                slowk_matype=0, 
                slowd_period=3, 
                slowd_matype=0
            )
            self.data['Stoch_K'] = slowk
            self.data['Stoch_D'] = slowd
            
            # Bollinger Bantları
            upper, middle, lower = talib.BBANDS(
                self.data['Close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2, 
                matype=0
            )
            self.data['BB_Upper'] = upper
            self.data['BB_Middle'] = middle
            self.data['BB_Lower'] = lower
            
            # ATR (Ortalama Gerçek Aralık)
            self.data['ATR'] = talib.ATR(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'], 
                timeperiod=14
            )
            
            # OBV (On-Balance Volume)
            self.data['OBV'] = talib.OBV(self.data['Close'], self.data['Volume'])
            
            # ADX (Ortalama Yön Endeksi)
            self.data['ADX'] = talib.ADX(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'], 
                timeperiod=14
            )
            
            # EMA (Üstel Hareketli Ortalama)
            self.data['EMA9'] = talib.EMA(self.data['Close'], timeperiod=9)
            self.data['EMA21'] = talib.EMA(self.data['Close'], timeperiod=21)
            self.data['EMA50'] = talib.EMA(self.data['Close'], timeperiod=50)
            self.data['EMA200'] = talib.EMA(self.data['Close'], timeperiod=200)
            
            # Ichimoku Cloud
            # Tenkan-sen (Dönüşüm Çizgisi)
            period9_high = self.data['High'].rolling(window=9).max()
            period9_low = self.data['Low'].rolling(window=9).min()
            self.data['Tenkan_Sen'] = (period9_high + period9_low) / 2
            
            # Kijun-sen (Taban Çizgisi)
            period26_high = self.data['High'].rolling(window=26).max()
            period26_low = self.data['Low'].rolling(window=26).min()
            self.data['Kijun_Sen'] = (period26_high + period26_low) / 2
            
            # Senkou Span A (Öncü Yayılma A)
            self.data['Senkou_Span_A'] = ((self.data['Tenkan_Sen'] + self.data['Kijun_Sen']) / 2).shift(26)
            
            # Senkou Span B (Öncü Yayılma B)
            period52_high = self.data['High'].rolling(window=52).max()
            period52_low = self.data['Low'].rolling(window=52).min()
            self.data['Senkou_Span_B'] = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span (Gecikmeli Yayılma)
            self.data['Chikou_Span'] = self.data['Close'].shift(-26)
            
            print("Teknik göstergeler başarıyla hesaplandı")
            return True
            
        except Exception as e:
            print(f"Teknik gösterge hesaplama hatası: {str(e)}")
            return False
    
    def calculate_fibonacci_levels(self, period=120):
        """
        Fibonacci seviyelerini hesaplar
        
        Parameters:
        -----------
        period : int, optional
            Hesaplama için kullanılacak dönem (gün sayısı)
        """
        if self.data is None:
            print("Önce veri yüklemelisiniz")
            return False
        
        try:
            # Son 'period' günlük veriyi al
            recent_data = self.data.iloc[-period:]
            
            # En yüksek ve en düşük değerleri bul
            high = recent_data['High'].max()
            low = recent_data['Low'].min()
            
            # Fibonacci seviyelerini hesapla
            for level in self.fibonacci_levels:
                level_name = f"Fib_{str(level).replace('.', '_')}"
                self.data[level_name] = low + (high - low) * level
            
            print("Fibonacci seviyeleri başarıyla hesaplandı")
            return True
            
        except Exception as e:
            print(f"Fibonacci seviyesi hesaplama hatası: {str(e)}")
            return False
    
    def calculate_gann_angles(self, start_price=None, start_date=None):
        """
        Gann açılarını hesaplar
        
        Parameters:
        -----------
        start_price : float, optional
            Başlangıç fiyatı
        start_date : datetime, optional
            Başlangıç tarihi
        """
        if self.data is None:
            print("Önce veri yüklemelisiniz")
            return False
        
        try:
            # Başlangıç değerlerini belirle
            if start_price is None:
                start_price = self.data['Close'].iloc[0]
            
            if start_date is None:
                start_date = self.data.index[0]
            
            # Her tarih için gün farkını hesapla
            self.data['Days'] = (self.data.index - start_date).days
            
            # Gann açılarını hesapla
            for x, y in self.gann_angles:
                angle_name = f"Gann_{x}x{y}"
                # Gann açısı formülü: Başlangıç fiyatı + (gün sayısı * x/y * birim fiyat)
                # Birim fiyat genellikle başlangıç fiyatının %1'i olarak alınır
                unit_price = start_price * 0.01
                self.data[angle_name] = start_price + (self.data['Days'] * (x/y) * unit_price)
            
            print("Gann açıları başarıyla hesaplandı")
            return True
            
        except Exception as e:
            print(f"Gann açısı hesaplama hatası: {str(e)}")
            return False
    
    def calculate_planetary_positions(self, date=None):
        """
        Belirli bir tarih için gezegen pozisyonlarını hesaplar
        
        Parameters:
        -----------
        date : datetime, optional
            Hesaplama yapılacak tarih
        
        Returns:
        --------
        dict
            Gezegen pozisyonları (burç dereceleri)
        """
        if date is None:
            date = datetime.now()
        
        # Ephem için tarih formatı
        ephem_date = ephem.Date(date)
        
        # Gözlem yeri (İstanbul)
        observer = ephem.Observer()
        observer.lat = '41.0082'  # İstanbul enlem
        observer.lon = '28.9784'  # İstanbul boylam
        observer.date = ephem_date
        
        planet_positions = {}
        
        for planet_name, planet in self.planets.items():
            planet.compute(observer)
            
            # Ekliptik koordinatları al (burç derecesi)
            ecliptic_lon = math.degrees(float(planet.hlong))
            
            # 0-360 derece aralığına normalize et
            ecliptic_lon = ecliptic_lon % 360
            
            planet_positions[planet_name] = ecliptic_lon
        
        return planet_positions
    
    def calculate_aspects(self, positions):
        """
        Gezegenler arası açıları hesaplar
        
        Parameters:
        -----------
        positions : dict
            Gezegen pozisyonları
        
        Returns:
        --------
        list
            Gezegenler arası açılar
        """
        aspects_list = []
        
        # Tüm gezegen çiftleri için açıları hesapla
        for p1 in positions:
            for p2 in positions:
                if p1 != p2:
                    # İki gezegen arasındaki açıyı hesapla
                    angle = abs(positions[p1] - positions[p2]) % 360
                    if angle > 180:
                        angle = 360 - angle
                    
                    # Açı türünü belirle (5 derece tolerans)
                    for aspect_name, aspect_angle in self.aspects.items():
                        if abs(angle - aspect_angle) <= 5:
                            aspects_list.append({
                                'Planet1': p1,
                                'Planet2': p2,
                                'Aspect': aspect_name,
                                'Angle': angle,
                                'Exact_Angle': aspect_angle
                            })
        
        return aspects_list
    
    def calculate_astro_score(self, date=None):
        """
        Astrolojik faktörlere dayalı piyasa skoru hesaplar
        
        Parameters:
        -----------
        date : datetime, optional
            Hesaplama yapılacak tarih
        
        Returns:
        --------
        float
            Astrolojik skor (-1 ile 1 arasında)
        """
        if date is None:
            date = datetime.now()
        
        # Gezegen pozisyonlarını hesapla
        positions = self.calculate_planetary_positions(date)
        
        # Açıları hesapla
        aspects = self.calculate_aspects(positions)
        
        # Başlangıç skoru
        score = 0
        
        # Açıların etkisini hesapla
        for aspect in aspects:
            p1 = aspect['Planet1']
            p2 = aspect['Planet2']
            aspect_type = aspect['Aspect']
            
            # Gezegen etkilerini al
            p1_influence = self.planet_influences.get(p1, 0.5)
            p2_influence = self.planet_influences.get(p2, 0.5)
            
            # Açı etkisi
            if aspect_type in ['Trine', 'Sextile']:
                # Olumlu açılar
                aspect_effect = 0.1
            elif aspect_type in ['Square', 'Opposition']:
                # Olumsuz açılar
                aspect_effect = -0.1
            elif aspect_type == 'Conjunction':
                # Kavuşum - gezegene göre değişir
                if p1 in ['Jupiter', 'Venus'] or p2 in ['Jupiter', 'Venus']:
                    aspect_effect = 0.1
                elif p1 in ['Saturn', 'Mars'] or p2 in ['Saturn', 'Mars']:
                    aspect_effect = -0.1
                else:
                    aspect_effect = 0.05
            else:
                # Diğer açılar
                aspect_effect = 0.02
            
            # Toplam etki
            score += aspect_effect * (p1_influence + p2_influence) / 2
        
        # Ay fazını hesapla
        moon = ephem.Moon()
        moon.compute(date)
        moon_phase = moon.phase / 100  # 0-1 arasında normalize et
        
        # Dolunay ve yeni ay etkileri
        if 0.45 < moon_phase < 0.55:  # Dolunay civarı
            score += 0.1
        elif moon_phase < 0.05 or moon_phase > 0.95:  # Yeni ay civarı
            score -= 0.1
        
        # Merkür retrosu kontrolü (varsa)
        mercury = ephem.Mercury()
        mercury.compute(date)
        if hasattr(mercury, 'retrograde') and mercury.retrograde:
            score -= 0.15
        
        # Skoru -1 ile 1 arasına normalize et
        score = max(min(score, 1), -1)
        
        return score
    
    def calculate_gann_cycle_position(self, date=None):
        """
        Gann döngülerindeki pozisyonu hesaplar
        
        Parameters:
        -----------
        date : datetime, optional
            Hesaplama yapılacak tarih
        
        Returns:
        --------
        dict
            Gann döngülerindeki pozisyonlar
        """
        if date is None:
            date = datetime.now()
        
        # Yıl başlangıcı
        year_start = datetime(date.year, 1, 1)
        
        # Yıl başlangıcından bugüne gün sayısı
        days_from_year_start = (date - year_start).days
        
        cycle_positions = {}
        
        # Her döngü için pozisyon hesapla
        for cycle in self.gann_cycles:
            position = days_from_year_start % cycle
            cycle_positions[f"{cycle}_Day_Cycle"] = position
        
        return cycle_positions
    
    def calculate_gann_score(self, date=None):
        """
        Gann teorisine dayalı piyasa skoru hesaplar
        
        Parameters:
        -----------
        date : datetime, optional
            Hesaplama yapılacak tarih
        
        Returns:
        --------
        float
            Gann skoru (-1 ile 1 arasında)
        """
        if date is None:
            date = datetime.now()
        
        # Gann döngü pozisyonlarını hesapla
        cycle_positions = self.calculate_gann_cycle_position(date)
        
        # Başlangıç skoru
        score = 0
        
        # 90 günlük döngü etkisi
        cycle_90_pos = cycle_positions.get('90_Day_Cycle', 0)
        if cycle_90_pos < 22:
            # İlk çeyrek - genellikle yükseliş
            score += 0.2
        elif cycle_90_pos < 45:
            # İkinci çeyrek - karışık
            score += 0.05
        elif cycle_90_pos < 67:
            # Üçüncü çeyrek - genellikle düşüş
            score -= 0.1
        else:
            # Son çeyrek - karışık, genellikle düşüş
            score -= 0.15
        
        # 45 günlük döngü etkisi
        cycle_45_pos = cycle_positions.get('45_Day_Cycle', 0)
        if cycle_45_pos < 22:
            # İlk yarı - genellikle yükseliş
            score += 0.1
        else:
            # İkinci yarı - genellikle düşüş
            score -= 0.1
        
        # 30 günlük döngü etkisi
        cycle_30_pos = cycle_positions.get('30_Day_Cycle', 0)
        if cycle_30_pos < 15:
            # İlk yarı - genellikle yükseliş
            score += 0.05
        else:
            # İkinci yarı - genellikle düşüş
            score -= 0.05
        
        # Gann açı etkisi
        # Bu kısım veri setine bağlı olduğu için burada hesaplanmıyor
        
        # Skoru -1 ile 1 arasına normalize et
        score = max(min(score, 1), -1)
        
        return score
    
    def calculate_technical_score(self, row):
        """
        Teknik göstergelere dayalı piyasa skoru hesaplar
        
        Parameters:
        -----------
        row : pandas.Series
            Veri satırı
        
        Returns:
        --------
        float
            Teknik skor (-1 ile 1 arasında)
        """
        score = 0
        
        # RSI
        if 'RSI' in row:
            rsi = row['RSI']
            if rsi < 30:
                score += 0.2  # Aşırı satım - alış sinyali
            elif rsi > 70:
                score -= 0.2  # Aşırı alım - satış sinyali
            elif rsi < 45:
                score += 0.1  # Satım bölgesi
            elif rsi > 55:
                score -= 0.1  # Alım bölgesi
        
        # MACD
        if 'MACD' in row and 'MACD_Signal' in row:
            macd = row['MACD']
            macd_signal = row['MACD_Signal']
            
            if macd > macd_signal:
                score += 0.15  # MACD sinyal çizgisinin üzerinde - alış sinyali
            else:
                score -= 0.15  # MACD sinyal çizgisinin altında - satış sinyali
        
        # Stokastik
        if 'Stoch_K' in row and 'Stoch_D' in row:
            k = row['Stoch_K']
            d = row['Stoch_D']
            
            if k < 20 and d < 20:
                score += 0.15  # Aşırı satım - alış sinyali
            elif k > 80 and d > 80:
                score -= 0.15  # Aşırı alım - satış sinyali
            
            if k > d:
                score += 0.1  # K, D'nin üzerinde - alış sinyali
            else:
                score -= 0.1  # K, D'nin altında - satış sinyali
        
        # Bollinger Bantları
        if 'Close' in row and 'BB_Upper' in row and 'BB_Lower' in row:
            close = row['Close']
            upper = row['BB_Upper']
            lower = row['BB_Lower']
            
            if close > upper:
                score -= 0.15  # Fiyat üst bandın üzerinde - aşırı alım
            elif close < lower:
                score += 0.15  # Fiyat alt bandın altında - aşırı satım
        
        # EMA
        if 'Close' in row and 'EMA9' in row and 'EMA21' in row and 'EMA50' in row and 'EMA200' in row:
            close = row['Close']
            ema9 = row['EMA9']
            ema21 = row['EMA21']
            ema50 = row['EMA50']
            ema200 = row['EMA200']
            
            # Kısa vadeli trend
            if close > ema9:
                score += 0.05
            else:
                score -= 0.05
            
            # Orta vadeli trend
            if close > ema21:
                score += 0.1
            else:
                score -= 0.1
            
            # Uzun vadeli trend
            if close > ema50:
                score += 0.15
            else:
                score -= 0.15
            
            # Çok uzun vadeli trend
            if close > ema200:
                score += 0.2
            else:
                score -= 0.2
        
        # Ichimoku Cloud
        if 'Close' in row and 'Tenkan_Sen' in row and 'Kijun_Sen' in row and 'Senkou_Span_A' in row and 'Senkou_Span_B' in row:
            close = row['Close']
            tenkan = row['Tenkan_Sen']
            kijun = row['Kijun_Sen']
            span_a = row['Senkou_Span_A']
            span_b = row['Senkou_Span_B']
            
            # Tenkan-Kijun Kesişimi
            if tenkan > kijun:
                score += 0.1  # Tenkan, Kijun'un üzerinde - alış sinyali
            else:
                score -= 0.1  # Tenkan, Kijun'un altında - satış sinyali
            
            # Bulut Pozisyonu
            if close > span_a and close > span_b:
                score += 0.15  # Fiyat bulutun üzerinde - yükseliş trendi
            elif close < span_a and close < span_b:
                score -= 0.15  # Fiyat bulutun altında - düşüş trendi
            
            # Bulut Rengi
            if span_a > span_b:
                score += 0.05  # Yeşil bulut - yükseliş trendi
            else:
                score -= 0.05  # Kırmızı bulut - düşüş trendi
        
        # Skoru -1 ile 1 arasına normalize et
        score = max(min(score, 1), -1)
        
        return score
    
    def generate_predictions(self, days=7, intervals_per_day=4):
        """
        Gelecek dönem için tahminler oluşturur
        
        Parameters:
        -----------
        days : int, optional
            Tahmin yapılacak gün sayısı
        intervals_per_day : int, optional
            Gün başına aralık sayısı
        
        Returns:
        --------
        pandas.DataFrame
            Tahmin sonuçları
        """
        if self.data is None:
            print("Önce veri yüklemelisiniz")
            return None
        
        try:
            # Son tarihi al
            last_date = self.data.index[-1]
            
            # Son fiyatı al
            last_close = self.data['Close'].iloc[-1]
            
            # Tahmin tarihleri oluştur
            prediction_dates = []
            for day in range(days):
                for interval in range(intervals_per_day):
                    hours_to_add = (day * 24) + (interval * (24 / intervals_per_day))
                    prediction_date = last_date + timedelta(hours=hours_to_add)
                    prediction_dates.append(prediction_date)
            
            # Tahmin sonuçları için DataFrame oluştur
            predictions = pd.DataFrame(index=prediction_dates)
            
            # Her tarih için tahmin yap
            for date in prediction_dates:
                # Astrolojik skor
                astro_score = self.calculate_astro_score(date)
                
                # Gann skoru
                gann_score = self.calculate_gann_score(date)
                
                # Teknik skor (son veri satırından)
                tech_score = self.calculate_technical_score(self.data.iloc[-1])
                
                # Ağırlıklı toplam skor
                total_score = (
                    astro_score * self.indicator_weights['Astro'] +
                    gann_score * self.indicator_weights['Gann'] +
                    tech_score * (1 - self.indicator_weights['Astro'] - self.indicator_weights['Gann'])
                )
                
                # Yön ve güç belirleme
                if total_score > 0.2:
                    direction = "Güçlü Yükseliş"
                    strength = abs(total_score) * 5
                elif total_score > 0.05:
                    direction = "Yükseliş"
                    strength = abs(total_score) * 5
                elif total_score > -0.05:
                    direction = "Yatay/Nötr"
                    strength = abs(total_score) * 5
                elif total_score > -0.2:
                    direction = "Düşüş"
                    strength = abs(total_score) * 5
                else:
                    direction = "Güçlü Düşüş"
                    strength = abs(total_score) * 5
                
                # Tahmini fiyat değişimi (yüzde olarak)
                price_change_pct = total_score * 0.5  # Puan başına %0.5 değişim
                
                # Tahmini fiyat
                estimated_price = last_close * (1 + price_change_pct)
                
                # Sonuçları kaydet
                predictions.at[date, 'Direction'] = direction
                predictions.at[date, 'Strength'] = f"{strength:.2f}/10"
                predictions.at[date, 'Estimated_Price'] = estimated_price
                predictions.at[date, 'Change_Pct'] = price_change_pct * 100
                predictions.at[date, 'Astro_Score'] = astro_score
                predictions.at[date, 'Gann_Score'] = gann_score
                predictions.at[date, 'Tech_Score'] = tech_score
                predictions.at[date, 'Total_Score'] = total_score
            
            self.predictions = predictions
            print(f"Tahminler başarıyla oluşturuldu. Toplam {len(predictions)} tahmin.")
            return predictions
            
        except Exception as e:
            print(f"Tahmin oluşturma hatası: {str(e)}")
            return None
    
    def plot_predictions(self, days=7):
        """
        Tahmin sonuçlarını görselleştirir
        
        Parameters:
        -----------
        days : int, optional
            Görselleştirilecek gün sayısı
        """
        if self.predictions is None:
            print("Önce tahmin oluşturmalısınız")
            return
        
        try:
            # Görselleştirilecek tahminleri filtrele
            if days < 7:
                end_date = self.predictions.index[0] + timedelta(days=days)
                plot_predictions = self.predictions.loc[:end_date]
            else:
                plot_predictions = self.predictions
            
            # Figür oluştur
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Fiyat grafiği
            ax1.plot(plot_predictions.index, plot_predictions['Estimated_Price'], 'b-', label='Tahmini Fiyat')
            
            # Yön ve güç gösterimi için renklendirme
            for i in range(len(plot_predictions) - 1):
                date = plot_predictions.index[i]
                next_date = plot_predictions.index[i + 1]
                direction = plot_predictions.loc[date, 'Direction']
                
                if 'Yükseliş' in direction:
                    color = 'green'
                elif 'Düşüş' in direction:
                    color = 'red'
                else:
                    color = 'gray'
                
                ax1.plot([date, next_date], 
                         [plot_predictions.loc[date, 'Estimated_Price'], 
                          plot_predictions.loc[next_date, 'Estimated_Price']], 
                         color=color, linewidth=2)
            
            # Skor grafiği
            ax2.plot(plot_predictions.index, plot_predictions['Astro_Score'], 'r-', label='Astrolojik Skor')
            ax2.plot(plot_predictions.index, plot_predictions['Gann_Score'], 'g-', label='Gann Skoru')
            ax2.plot(plot_predictions.index, plot_predictions['Tech_Score'], 'b-', label='Teknik Skor')
            ax2.plot(plot_predictions.index, plot_predictions['Total_Score'], 'k-', linewidth=2, label='Toplam Skor')
            
            # Sıfır çizgisi
            ax2.axhline(y=0, color='gray', linestyle='--')
            
            # Grafik ayarları
            ax1.set_title('AstroGann İndikatörü - Fiyat Tahminleri')
            ax1.set_ylabel('Fiyat')
            ax1.legend()
            ax1.grid(True)
            
            ax2.set_title('Tahmin Skorları')
            ax2.set_xlabel('Tarih')
            ax2.set_ylabel('Skor')
            ax2.set_ylim(-1, 1)
            ax2.legend()
            ax2.grid(True)
            
            # X ekseni tarih formatı
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Grafik oluşturma hatası: {str(e)}")
    
    def export_predictions(self, file_path):
        """
        Tahmin sonuçlarını dışa aktarır
        
        Parameters:
        -----------
        file_path : str
            Dışa aktarılacak dosya yolu
        """
        if self.predictions is None:
            print("Önce tahmin oluşturmalısınız")
            return False
        
        try:
            # CSV olarak dışa aktar
            self.predictions.to_csv(file_path)
            print(f"Tahminler başarıyla dışa aktarıldı: {file_path}")
            return True
            
        except Exception as e:
            print(f"Dışa aktarma hatası: {str(e)}")
            return False
    
    def run_gui(self):
        """
        Grafiksel kullanıcı arayüzünü başlatır
        """
        # Ana pencere
        root = tk.Tk()
        root.title(f"AstroGann İndikatörü v{self.version}")
        root.geometry("1200x800")
        
        # Stil ayarları
        style = ttk.Style()
        style.theme_use('clam')
        
        # Ana çerçeve
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Üst panel - Veri yükleme ve işleme
        top_frame = ttk.LabelFrame(main_frame, text="Veri Yükleme ve İşleme", padding=10)
        top_frame.pack(fill=tk.X, pady=5)
        
        # Veri yükleme
        load_frame = ttk.Frame(top_frame)
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(load_frame, text="Veri Kaynağı:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        data_source = tk.StringVar(value="csv")
        ttk.Radiobutton(load_frame, text="CSV Dosyası", variable=data_source, value="csv").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(load_frame, text="API", variable=data_source, value="api").grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(load_frame, text="Sembol:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        symbol_entry = ttk.Entry(load_frame, width=15)
        symbol_entry.grid(row=1, column=1, padx=5, pady=5)
        symbol_entry.insert(0, "BTCUSDT")
        
        ttk.Label(load_frame, text="Dosya:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        file_path_var = tk.StringVar()
        file_path_entry = ttk.Entry(load_frame, textvariable=file_path_var, width=40)
        file_path_entry.grid(row=1, column=3, padx=5, pady=5)
        
        def browse_file():
            file_path = filedialog.askopenfilename(filetypes=[("CSV Dosyaları", "*.csv"), ("Tüm Dosyalar", "*.*")])
            if file_path:
                file_path_var.set(file_path)
        
        browse_button = ttk.Button(load_frame, text="Gözat", command=browse_file)
        browse_button.grid(row=1, column=4, padx=5, pady=5)
        
        # Data preview table for loaded CSV
        data_preview_frame = ttk.Frame(top_frame)
        data_preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.data_tree = ttk.Treeview(data_preview_frame, columns=('Date','Open','High','Low','Close','Volume'), show='headings')
        for col in self.data_tree['columns']:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=80, minwidth=80, anchor=tk.CENTER, stretch=True)
        vsb = ttk.Scrollbar(data_preview_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb = ttk.Scrollbar(data_preview_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(xscrollcommand=hsb.set)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_tree.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(load_frame, text="Başlangıç Tarihi:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        start_date_entry = ttk.Entry(load_frame, width=15)
        start_date_entry.grid(row=2, column=1, padx=5, pady=5)
        start_date_entry.insert(0, (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        
        ttk.Label(load_frame, text="Bitiş Tarihi:").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        end_date_entry = ttk.Entry(load_frame, width=15)
        end_date_entry.grid(row=2, column=3, padx=5, pady=5)
        end_date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))
        
        # İşlem butonları
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def load_data_action():
            try:
                source = data_source.get()
                symbol = symbol_entry.get()
                file_path = file_path_var.get()
                start_date = start_date_entry.get()
                end_date = end_date_entry.get()
                # If no CSV selected, use default example_data.csv in current directory
                if source == "csv" and not file_path:
                    default_path = os.path.join(os.getcwd(), 'example_data.csv')
                    file_path = default_path
                    file_path_var.set(file_path)
                
                if source == "csv":
                    if not file_path:
                        messagebox.showerror("Hata", "Lütfen bir CSV dosyası seçin")
                        return
                    success = self.load_data(file_path=file_path, source=source)
                else:
                    if not symbol:
                        messagebox.showerror("Hata", "Lütfen bir sembol girin")
                        return
                    success = self.load_data(symbol=symbol, start_date=start_date, end_date=end_date, source=source)
                
                if success:
                    messagebox.showinfo("Başarılı", f"Veri başarıyla yüklendi. Toplam {len(self.data)} kayıt.")
                    status_var.set(f"Veri yüklendi: {len(self.data)} kayıt")
                    
                    # Veri yüklendikten sonra diğer butonları etkinleştir
                    calc_indicators_button.config(state=tk.NORMAL)
                    calc_fibonacci_button.config(state=tk.NORMAL)
                    calc_gann_button.config(state=tk.NORMAL)
                    generate_predictions_button.config(state=tk.NORMAL)
                    
                    # Populate preview table
                    for item in self.data_tree.get_children():
                        self.data_tree.delete(item)
                    for idx, row in self.data.iterrows():
                        self.data_tree.insert('', tk.END, values=(
                            idx.strftime('%Y-%m-%d'),
                            row['Open'], row['High'], row['Low'], row['Close'], row['Volume']
                        ))
                else:
                    messagebox.showerror("Hata", "Veri yükleme başarısız")
            except Exception as e:
                messagebox.showerror("Hata", f"Veri yükleme hatası: {str(e)}")
        
        def calculate_indicators_action():
            try:
                success = self.calculate_technical_indicators()
                if success:
                    messagebox.showinfo("Başarılı", "Teknik göstergeler başarıyla hesaplandı")
                    status_var.set("Teknik göstergeler hesaplandı")
                    # Refresh preview columns to include new indicators
                    new_cols = ['Date'] + list(self.data.columns)
                    self.data_tree.config(columns=new_cols)
                    for col in new_cols:
                        self.data_tree.heading(col, text=col)
                        self.data_tree.column(col, width=100)
                    # Repopulate preview table with updated data
                    for item in self.data_tree.get_children():
                        self.data_tree.delete(item)
                    for idx, row in self.data.iterrows():
                        values = [idx.strftime('%Y-%m-%d')] + [row[col] for col in self.data.columns]
                        self.data_tree.insert('', tk.END, values=values)
                else:
                    messagebox.showerror("Hata", "Teknik gösterge hesaplama başarısız")
            except Exception as e:
                messagebox.showerror("Hata", f"Teknik gösterge hesaplama hatası: {str(e)}")
        
        def calculate_fibonacci_action():
            try:
                period = int(fibonacci_period_entry.get())
                success = self.calculate_fibonacci_levels(period=period)
                if success:
                    messagebox.showinfo("Başarılı", "Fibonacci seviyeleri başarıyla hesaplandı")
                    status_var.set("Fibonacci seviyeleri hesaplandı")
                    # Refresh preview columns to include Fibonacci levels
                    new_cols = ['Date'] + list(self.data.columns)
                    self.data_tree.config(columns=new_cols)
                    for col in new_cols:
                        self.data_tree.heading(col, text=col)
                        self.data_tree.column(col, width=100)
                    # Repopulate preview table with updated data
                    for item in self.data_tree.get_children():
                        self.data_tree.delete(item)
                    for idx, row in self.data.iterrows():
                        values = [idx.strftime('%Y-%m-%d')] + [row[col] for col in self.data.columns]
                        self.data_tree.insert('', tk.END, values=values)
                else:
                    messagebox.showerror("Hata", "Fibonacci seviyesi hesaplama başarısız")
            except Exception as e:
                messagebox.showerror("Hata", f"Fibonacci seviyesi hesaplama hatası: {str(e)}")
        
        def calculate_gann_action():
            try:
                success = self.calculate_gann_angles()
                if success:
                    messagebox.showinfo("Başarılı", "Gann açıları başarıyla hesaplandı")
                    status_var.set("Gann açıları hesaplandı")
                    # Refresh preview columns to include Gann angles
                    new_cols = ['Date'] + list(self.data.columns)
                    self.data_tree.config(columns=new_cols)
                    for col in new_cols:
                        self.data_tree.heading(col, text=col)
                        self.data_tree.column(col, width=100)
                    # Repopulate preview table with updated data
                    for item in self.data_tree.get_children():
                        self.data_tree.delete(item)
                    for idx, row in self.data.iterrows():
                        values = [idx.strftime('%Y-%m-%d')] + [row[col] for col in self.data.columns]
                        self.data_tree.insert('', tk.END, values=values)
                else:
                    messagebox.showerror("Hata", "Gann açısı hesaplama başarısız")
            except Exception as e:
                messagebox.showerror("Hata", f"Gann açısı hesaplama hatası: {str(e)}")
        
        def generate_predictions_action():
            try:
                days = int(prediction_days_entry.get())
                intervals = int(prediction_intervals_entry.get())
                
                predictions = self.generate_predictions(days=days, intervals_per_day=intervals)
                if predictions is not None:
                    messagebox.showinfo("Başarılı", f"Tahminler başarıyla oluşturuldu. Toplam {len(predictions)} tahmin.")
                    status_var.set(f"Tahminler oluşturuldu: {len(predictions)} tahmin")
                    
                    # Tahminler oluşturulduktan sonra diğer butonları etkinleştir
                    plot_predictions_button.config(state=tk.NORMAL)
                    export_predictions_button.config(state=tk.NORMAL)
                    
                    # Tahmin sonuçlarını göster
                    show_predictions()
                else:
                    messagebox.showerror("Hata", "Tahmin oluşturma başarısız")
            except Exception as e:
                messagebox.showerror("Hata", f"Tahmin oluşturma hatası: {str(e)}")
        
        def plot_predictions_action():
            try:
                days = int(prediction_days_entry.get())
                self.plot_predictions(days=days)
            except Exception as e:
                messagebox.showerror("Hata", f"Grafik oluşturma hatası: {str(e)}")
        
        def export_predictions_action():
            try:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV Dosyaları", "*.csv"), ("Tüm Dosyalar", "*.*")]
                )
                if file_path:
                    success = self.export_predictions(file_path)
                    if success:
                        messagebox.showinfo("Başarılı", f"Tahminler başarıyla dışa aktarıldı: {file_path}")
                    else:
                        messagebox.showerror("Hata", "Dışa aktarma başarısız")
            except Exception as e:
                messagebox.showerror("Hata", f"Dışa aktarma hatası: {str(e)}")
        
        load_data_button = ttk.Button(button_frame, text="Veri Yükle", command=load_data_action)
        load_data_button.pack(side=tk.LEFT, padx=5)
        
        calc_indicators_button = ttk.Button(button_frame, text="Teknik Göstergeleri Hesapla", command=calculate_indicators_action, state=tk.DISABLED)
        calc_indicators_button.pack(side=tk.LEFT, padx=5)
        
        # Fibonacci periyodu
        ttk.Label(button_frame, text="Fibonacci Periyodu:").pack(side=tk.LEFT, padx=5)
        fibonacci_period_entry = ttk.Entry(button_frame, width=5)
        fibonacci_period_entry.pack(side=tk.LEFT)
        fibonacci_period_entry.insert(0, "120")
        
        calc_fibonacci_button = ttk.Button(button_frame, text="Fibonacci Hesapla", command=calculate_fibonacci_action, state=tk.DISABLED)
        calc_fibonacci_button.pack(side=tk.LEFT, padx=5)
        
        calc_gann_button = ttk.Button(button_frame, text="Gann Açılarını Hesapla", command=calculate_gann_action, state=tk.DISABLED)
        calc_gann_button.pack(side=tk.LEFT, padx=5)
        
        # Orta panel - Tahmin ayarları
        middle_frame = ttk.LabelFrame(main_frame, text="Tahmin Ayarları", padding=10)
        middle_frame.pack(fill=tk.X, pady=5)
        
        prediction_frame = ttk.Frame(middle_frame)
        prediction_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(prediction_frame, text="Tahmin Günü:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        prediction_days_entry = ttk.Entry(prediction_frame, width=5)
        prediction_days_entry.grid(row=0, column=1, padx=5, pady=5)
        prediction_days_entry.insert(0, "7")
        
        ttk.Label(prediction_frame, text="Gün Başına Aralık:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        prediction_intervals_entry = ttk.Entry(prediction_frame, width=5)
        prediction_intervals_entry.grid(row=0, column=3, padx=5, pady=5)
        prediction_intervals_entry.insert(0, "4")
        
        prediction_button_frame = ttk.Frame(middle_frame)
        prediction_button_frame.pack(fill=tk.X, pady=5)
        
        generate_predictions_button = ttk.Button(prediction_button_frame, text="Tahmin Oluştur", command=generate_predictions_action, state=tk.DISABLED)
        generate_predictions_button.pack(side=tk.LEFT, padx=5)
        
        plot_predictions_button = ttk.Button(prediction_button_frame, text="Tahminleri Görselleştir", command=plot_predictions_action, state=tk.DISABLED)
        plot_predictions_button.pack(side=tk.LEFT, padx=5)
        
        export_predictions_button = ttk.Button(prediction_button_frame, text="Tahminleri Dışa Aktar", command=export_predictions_action, state=tk.DISABLED)
        export_predictions_button.pack(side=tk.LEFT, padx=5)
        
        # Alt panel - Sonuçlar
        bottom_frame = ttk.LabelFrame(main_frame, text="Tahmin Sonuçları", padding=10)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Sonuç tablosu
        columns = ('Tarih', 'Yön', 'Güç', 'Tahmini Fiyat', 'Değişim (%)', 'Astro Skoru', 'Gann Skoru', 'Teknik Skor', 'Toplam Skor')
        result_tree = ttk.Treeview(bottom_frame, columns=columns, show='headings')
        
        # Sütun başlıkları
        for col in columns:
            result_tree.heading(col, text=col)
            result_tree.column(col, width=120, minwidth=80, anchor=tk.CENTER, stretch=True)
        
        result_tree.pack(fill=tk.BOTH, expand=True)
        
        # Kaydırma çubukları
        scrollbar_y = ttk.Scrollbar(result_tree, orient=tk.VERTICAL, command=result_tree.yview)
        result_tree.configure(yscrollcommand=scrollbar_y.set)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = ttk.Scrollbar(bottom_frame, orient=tk.HORIZONTAL, command=result_tree.xview)
        result_tree.configure(xscrollcommand=scrollbar_x.set)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        def show_predictions():
            # Mevcut sonuçları temizle
            for item in result_tree.get_children():
                result_tree.delete(item)
            
            # Yeni sonuçları ekle
            if self.predictions is not None:
                for date, row in self.predictions.iterrows():
                    result_tree.insert('', tk.END, values=(
                        date.strftime('%Y-%m-%d %H:%M'),
                        row['Direction'],
                        row['Strength'],
                        f"{row['Estimated_Price']:.2f}",
                        f"{row['Change_Pct']:.2f}",
                        f"{row['Astro_Score']:.4f}",
                        f"{row['Gann_Score']:.4f}",
                        f"{row['Tech_Score']:.4f}",
                        f"{row['Total_Score']:.4f}"
                    ))
        
        # Durum çubuğu
        status_frame = ttk.Frame(root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        status_var = tk.StringVar(value="Hazır")
        status_label = ttk.Label(status_frame, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X)
        
        # Auto-load default CSV shortly after GUI starts
        default_csv = os.path.join(os.getcwd(), 'example_data.csv')
        if os.path.isfile(default_csv):
            file_path_var.set(default_csv)
            # Schedule automatic load after GUI is running
            root.after(100, load_data_action)

        # Pencereyi göster
        root.mainloop()

# Örnek kullanım
if __name__ == "__main__":
    indicator = AstroGannIndicator()
    indicator.run_gui()
