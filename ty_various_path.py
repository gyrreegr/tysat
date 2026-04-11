import pandas as pd
import numpy as np
import os
import requests
import urllib3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# 關閉因為略過 HTTPS 驗證而產生的 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. 檔案路徑與基礎設定 (請依據環境修改)
# ==========================================
OUTPUT_DIR = "./"
CSV_FILENAME = "watch_tracks.csv"
LOGO_PATH = "./logo.png"            
COUNTY_SHP =  "./COUNTY_MOI_1130718.shp"       # 縣市 SHP 檔案
SEA_WARNING_SHP = "./100km.shp"          # 海上警戒線 SHP 檔案

# 字體設定 (根據你的作業系統調整中文字體)
import matplotlib.font_manager as fm

FONT_PATH = "./font.ttf"   # 或 GitHub Actions 用相對路徑

if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    # fallback（避免 crash）
    plt.rcParams['font.sans-serif'] = [
        'Noto Sans CJK TC',
        'DejaVu Sans'
    ]


class TyphoonMapGenerator:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
        
        # 設定各國預報中心的顏色 + 觀測實線
        self.forecast_colors = {
            'obs': 'black',        # 觀測資料 - 黑實線
            'cwb': '#FF0000',      # 中央氣象署 - 紅色
            'jtwc': '#0000FF',     # 美國聯合颱風警報中心 - 藍色
            'jma': '#FF8000',      # 日本氣象廳 - 橙色
            'kma': '#800080',      # 韓國氣象廳 - 紫色
            'nmc': '#008000',      # 中國國家氣象中心 - 綠色
            'hko': '#FF69B4'       # 香港天文台 - 粉紅色
        }

    def download_data(self, output_dir='.'):
        """從 NCDR 網站互動式下載颱風路徑資料"""
        base_url = "https://watch.ncdr.nat.gov.tw/wh/cv_typhoon_tracks"
        output_path = os.path.join(output_dir, CSV_FILENAME)
        os.makedirs(output_dir, exist_ok=True)

        choice = "N"
        date_str, time_str = None, None

        if choice == 'Y':
            while True:
                try:
                    date_input = input("請輸入日期 (格式 YYYY-MM-DD): ")
                    datetime.strptime(date_input, '%Y-%m-%d')
                    date_str = date_input
                    break
                except ValueError:
                    print("日期格式錯誤，請重新輸入。")
            
            while True:
                try:
                    hour_input = int(input("請輸入整點小時 (0-23): "))
                    if 0 <= hour_input <= 23:
                        time_str = f"{hour_input:02d}:00"
                        break
                    else:
                        print("小時必須在 0 到 23 之間。")
                except ValueError:
                    print("請輸入有效的數字。")
            
            params = {'d': date_str, 't': time_str}
            print(f"正在嘗試下載指定時間 {date_str} {time_str} 的資料...")
            try:
                response = requests.get(base_url, params=params, timeout=15, verify=False)
                if response.status_code == 200 and 'TYPH_ID' in response.text:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ 資料成功下載並儲存至: {output_path}")
                    return output_path
                else:
                    print(f"❌ 下載失敗或該時間點無資料 (狀態碼: {response.status_code})。")
                    return None
            except requests.RequestException as e:
                print(f"❌ 下載時發生網路錯誤: {e}")
                return None
        else:
            now = datetime.now()
            for i in range(3):
                target_time = now - timedelta(hours=i)
                date_str = target_time.strftime('%Y-%m-%d')
                time_str = target_time.strftime('%H:00')
                
                params = {'d': date_str, 't': time_str}
                print(f"正在嘗試下載時間 {date_str} {time_str} 的資料... (第 {i+1} 次嘗試)")
                
                try:
                    response = requests.get(base_url, params=params, timeout=15, verify=False)
                    if response.status_code == 200 and 'TYPH_ID' in response.text and len(response.content) > 100:
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        print(f"✅ 資料成功下載並儲存至: {output_path}")
                        return output_path
                    else:
                        print("此時間點無資料或資料不完整，嘗試前一個小時...")
                except requests.RequestException as e:
                    print(f"❌ 下載時發生網路錯誤: {e}")
            
            print("❌ 錯誤：在最近3小時內均未找到有效的颱風資料。")
            return None
    
    def load_data(self, csv_file=None):
        """載入並清理颱風資料"""
        if csv_file is None:
            csv_file = self.csv_path
        
        print(f"載入檔案: {csv_file}")
        df = pd.read_csv(csv_file)
        df = df.replace([-999, -9999], np.nan)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])
        df = df.dropna(subset=['LON', 'LAT'])
        df = df[(df['LON'] >= -180) & (df['LON'] <= 180)]
        df = df[(df['LAT'] >= -90) & (df['LAT'] <= 90)]
        
        # 萃取各國中心代碼 (例如 2305_cwb_... 取出 cwb)
        df['CENTER'] = df['TYPH_ID'].str.split('_').str[1]
        return df

    def plot_map(self, df, extent, output_filename):
        """繪製單張地圖"""
        lon_min, lon_max, lat_min, lat_max = extent

        def in_bounds(x, y):
            """判斷點是否在畫面範圍內"""
            return lon_min <= x <= lon_max and lat_min <= y <= lat_max

        # --- A. 設定標題時間 ---
        obs_data = df[df['CENTER'] == 'obs']
        if not obs_data.empty:
            title_dt = obs_data['DATETIME'].max()  # 取最後一筆觀測資料的時間當基準
        else:
            title_dt = df['DATETIME'].min()
        title_time_str = title_dt.strftime("%Y年%m月%d日%H時")

        # --- B. 建立地圖圖層 ---
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # (Z-order 0) 海洋
        ax.add_feature(cfeature.OCEAN, facecolor='#235673', zorder=0)
        
        # (Z-order 1) 海上警戒線 SHP
        try:
            if os.path.exists(SEA_WARNING_SHP):
                sea_warn = gpd.read_file(SEA_WARNING_SHP)
                sea_warn.plot(ax=ax, color='#ffffff', alpha=0.15, zorder=1) 
        except Exception as e:
            print(f"讀取海警線 SHP 失敗: {e}")

        # (Z-order 2) 全球陸地
        ax.add_feature(cfeature.LAND, facecolor='white', zorder=2)
        
        # (Z-order 3) 縣市 SHP
        try:
            if os.path.exists(COUNTY_SHP):
                counties = gpd.read_file(COUNTY_SHP)
                counties.plot(ax=ax, facecolor='#d4b439', edgecolor='black', linewidth=1.2, zorder=3)
        except Exception as e:
            print(f"讀取縣市 SHP 失敗: {e}")

        # --- C. 畫路徑線與點 ---
        centers = df['CENTER'].unique()
        
        # 確保觀測(obs)最先畫，讓預報線疊在上面
        if 'obs' in centers:
            centers = list(centers)
            centers.remove('obs')
            centers.insert(0, 'obs')

        for center in centers:
            if center not in self.forecast_colors:
                continue
            
            center_data = df[df['CENTER'] == center]
            if center_data.empty:
                continue
                
            color = self.forecast_colors[center]
            
            # 依據 TYPH_ID 分組，確保不同批次/不同颱風的路徑各自獨立繪製
            for typh_id, c_data in center_data.groupby('TYPH_ID'):
                c_data = c_data.sort_values('DATETIME')
                lons = c_data['LON'].tolist()
                lats = c_data['LAT'].tolist()

                # (Z-order 6) 畫路線 (統一粗細)
                if len(lons) >= 2:
                    ax.plot(lons, lats, color=color, linestyle='-', linewidth=1, zorder=6)

                # (Z-order 7) 畫資料點
                valid_lons, valid_lats = [], []
                total_points = len(lons)
                
                for i, (lon, lat) in enumerate(zip(lons, lats)):
                    if in_bounds(lon, lat):
                        # 💡 關鍵修正：如果資料點太多(例如CWB的121點)，只取間隔畫圓圈
                        # 保留起點、終點，以及每 12 個點畫一次，避免擠成毛毛蟲
                        if total_points > 20:
                            if i == 0 or i == total_points - 1 or i % 12 == 0:
                                valid_lons.append(lon)
                                valid_lats.append(lat)
                        else:
                            # 點數少的國家(如 JMA, JTWC)就每個點都畫
                            valid_lons.append(lon)
                            valid_lats.append(lat)
                
                if valid_lons:
                    ax.scatter(valid_lons, valid_lats, color=color, s=15, zorder=7)

        # --- D. 畫左上角商標 ---
        try:
            if os.path.exists(LOGO_PATH):
                logo_img = mpimg.imread(LOGO_PATH)
                logo_box = OffsetImage(logo_img, zoom=0.03) 
                ab_logo = AnnotationBbox(logo_box, xy=(0.02, 0.98), xycoords='axes fraction', 
                                     box_alignment=(0, 1), frameon=False, zorder=10)
                ax.add_artist(ab_logo)
        except Exception as e:
            print(f"讀取商標失敗: {e}")

        # --- E. 標題與輸出 ---
        ax.set_title(f"各國預測路徑 ({title_time_str})", fontsize=18, pad=15, fontweight='bold')
        ax.outline_patch.set_visible(False) if hasattr(ax, 'outline_patch') else ax.spines['geo'].set_visible(False)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✅ 圖片已成功儲存至: {output_filename}")
        plt.close(fig)

    def run(self, output_dir):
        """主執行邏輯"""
        try:
            df = self.load_data()
            print(f"成功載入 {len(df)} 筆資料")
            print("\n--- 開始繪製地圖 ---")
            
            # 一般版
            normal_path = os.path.join(output_dir, "multi_country_path_normal.png")
            print("正在繪製一般版地圖...")
            self.plot_map(df, extent=[110, 170, 5, 35], output_filename=normal_path)
            
            # 放大版
            magnified_path = os.path.join(output_dir, "multi_country_path_magnified.png")
            print("正在繪製放大版地圖...")
            self.plot_map(df, extent=[115, 130, 15, 30], output_filename=magnified_path)
            
        except Exception as e:
            print(f"執行時發生錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 步驟一：下載資料
    downloader = TyphoonMapGenerator()
    INPUT_DIR = "./"
    csv_path = downloader.download_data(output_dir=INPUT_DIR)

    # 步驟二：載入資料並繪製地圖
    if csv_path:
        generator = TyphoonMapGenerator(csv_path=csv_path)
        generator.run(output_dir=OUTPUT_DIR)
        print("\n--- 程式執行完畢 ---")
    else:
        print("\n❌ 錯誤：因資料下載失敗，無法進行繪製，程式已終止。")