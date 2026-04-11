import json
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from datetime import datetime, timedelta
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import requests
import urllib3

# 關閉因為略過 HTTPS 驗證而產生的 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_cwa_json(url, save_path):
    """自動下載 CWA JSON 資料 (略過 HTTPS 驗證)"""
    try:
        print(f"📥 正在下載最新颱風資料...\n來源: {url}")
        # 加入 verify=False 來略過 SSL 憑證檢查
        response = requests.get(url, verify=False)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ 資料已成功下載至: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 下載失敗: {e}")

# ==========================================
# 1. 檔案路徑與基礎設定 (請依據你的環境修改)
# ==========================================
JSON_PATH = "./W-C0034-005.json"            # 颱風 JSON 資料
ICON_PATH = "./Typhoon point icon.png"            # 颱風位置點的自訂 Icon
COUNTY_SHP =  "./COUNTY_MOI_1130718.shp"            # 縣市 SHP 檔案
TOWNSHIP_SHP = "./TOWN_MOI_1111118.shp"     
LOGO_PATH = "./logo.png"  # 🆕 你的商標 PNG 檔案路徑
SEA_WARNING_SHP = "./100km.shp" # 海上警戒線 GeoJSON 檔案
OUTPUT_IMAGE = "typhoon_path_map.png"     # 輸出圖片名稱

# 字體設定 (根據你的作業系統調整中文字體，Windows 通常用 Microsoft JhengHei)
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

plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 地理運算與時間解析函數
# ==========================================
def get_wind_circle_polygon(lat, lon, radius_km):
    """將經緯度加上半徑(公里)產生暴風圈的 Shapely Polygon"""
    proj_wgs84 = pyproj.CRS('EPSG:4326')
    proj_merc = pyproj.CRS('EPSG:3857')
    transformer = pyproj.Transformer.from_crs(proj_wgs84, proj_merc, always_xy=True)
    transformer_back = pyproj.Transformer.from_crs(proj_merc, proj_wgs84, always_xy=True)
    
    x, y = transformer.transform(lon, lat)
    circle = Point(x, y).buffer(radius_km * 1000)
    return transform(transformer_back.transform, circle)

def parse_iso_time(iso_str):
    """解析 ISO 時間為 datetime 物件"""
    return datetime.fromisoformat(iso_str.replace('+08:00', ''))

def format_time_label(dt):
    """格式化為 ??日??時"""
    return dt.strftime("%d日%H時")

# ==========================================
# 3. 主程式
# ==========================================


def plot_typhoon_map(extent, output_filename):
    # 解析傳入的經緯度範圍
    lon_min, lon_max, lat_min, lat_max = extent

    # 建立一個小函數：判斷點是否在目前的經緯度範圍內
    def in_bounds(x, y):
        return lon_min <= x <= lon_max and lat_min <= y <= lat_max

    # --- A. 讀取 JSON 資料 ---
    with open(JSON_PATH, encoding='utf-8') as f:
        data = json.load(f)
        
    dataset = data['cwaopendata'].get('Dataset', data['cwaopendata'].get('dataset'))
    tc = dataset['TropicalCyclones']['TropicalCyclone']
    
    # 解析觀測 (Analysis) 與預報 (Forecast)
    analysis_fixes = tc.get('AnalysisData', {}).get('Fix', [])
    forecast_fixes = tc.get('ForecastData', {}).get('Fix', [])
    
    # 紀錄所有要畫的點 [lon, lat, label_text, is_forecast, fix_data, datetime_obj]
    points_to_draw = []
    
    # 處理觀測資料
    last_analysis_dt = None
    for fix in analysis_fixes:
        lon = float(fix['CoordinateLongitude'])
        lat = float(fix['CoordinateLatitude'])
        dt = parse_iso_time(fix['DateTime'])
        last_analysis_dt = dt
        points_to_draw.append([lon, lat, format_time_label(dt), False, fix, dt])
        
    # 處理預報資料
    for fix in forecast_fixes:
        lon = float(fix['CoordinateLongitude'])
        lat = float(fix['CoordinateLatitude'])
        init_dt = parse_iso_time(fix['InitialTime'])
        tau = int(fix['ForecastHour'])
        target_dt = init_dt + timedelta(hours=tau)
        points_to_draw.append([lon, lat, format_time_label(target_dt), True, fix, target_dt])

    # 取得最後標題時間
    if last_analysis_dt:
        title_time_str = last_analysis_dt.strftime("%Y年%m月%d日%H時")
    else:
        title_time_str = "未知時間"

    # --- B. 建立地圖與圖層 ---
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # 設定經緯度範圍
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
            counties.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2, zorder=4)
    except Exception as e:
        print(f"讀取縣市 SHP 失敗: {e}")

    # (Z-order 4) 鄉鎮 SHP
    try:
        if os.path.exists(TOWNSHIP_SHP):
            townships = gpd.read_file(TOWNSHIP_SHP)
            townships.plot(ax=ax, facecolor='#d4b439', edgecolor='darkgray', linewidth=0.5, zorder=3)
    except Exception as e:
        print(f"讀取鄉鎮 SHP 失敗: {e}")

    # --- C. 繪製颱風資訊 ---
    analysis_coords = [[p[0], p[1]] for p in points_to_draw if not p[3]]
    forecast_coords = [[p[0], p[1]] for p in points_to_draw if p[3]]

    # (Z-order 6) 畫路線
    # 💡 這裡不加 in_bounds 判斷，因為我們要把整條線丟給 Cartopy，
    # 它會自動把超出的部分切掉，並完美保留且畫出在經緯度範圍內的線段！
    if len(analysis_coords) >= 2:
        lons, lats = zip(*analysis_coords)
        ax.plot(lons, lats, color='black', linestyle='-', linewidth=2, zorder=6)
        
    if forecast_coords:
        if analysis_coords:
            fcst_line = [analysis_coords[-1]] + forecast_coords
        else:
            fcst_line = forecast_coords
        lons, lats = zip(*fcst_line)
        ax.plot(lons, lats, color='#ffa008', linestyle='-', linewidth=2, zorder=6)

    # (Z-order 5) 畫暴風圈
    for p in points_to_draw:
        lon, lat, _, _, fix, _ = p
        
        # 💡 新增條件：颱風中心必須在顯示範圍內才繪製暴風圈面
        if in_bounds(lon, lat):
            if fix.get('Circle25ms') and fix['Circle25ms'].get('Radius'):
                r25 = float(fix['Circle25ms']['Radius'])
                poly = get_wind_circle_polygon(lat, lon, r25)
                ax.add_geometries([poly], crs=ccrs.PlateCarree(), facecolor='#836040', alpha=0.4, edgecolor='none', zorder=5)
                
            if fix.get('Circle15ms') and fix['Circle15ms'].get('Radius'):
                r15 = float(fix['Circle15ms']['Radius'])
                poly = get_wind_circle_polygon(lat, lon, r15)
                ax.add_geometries([poly], crs=ccrs.PlateCarree(), facecolor='#5b8152', alpha=0.4, edgecolor='none', zorder=5)

    # (Z-order 7 & 8) 畫 Icon 與文字標籤
    try:
        icon_img = mpimg.imread(ICON_PATH)
        imagebox = OffsetImage(icon_img, zoom=0.015) 
    except Exception as e:
        print(f"讀取 Icon 失敗，將以普通點代替: {e}")
        imagebox = None

    labeled_dates = set()
    for p in points_to_draw:
        # 💡 注意這裡的解包：把第四個參數 is_forecast 拿出來用
        lon, lat, label, is_forecast, fix, dt = p
        
        if in_bounds(lon, lat):
            # 放置 Icon (無論是觀測還是預報，Icon 都要畫)
            if imagebox:
                ab = AnnotationBbox(imagebox, (lon, lat), frameon=False, zorder=7)
                ax.add_artist(ab)
            else:
                ax.scatter(lon, lat, color='red', zorder=7)
                
            # 💡 標示文字邏輯：新增 if is_forecast 判斷，只有預報資料才輸出文字
            if is_forecast:
                current_date = dt.date()
                if current_date not in labeled_dates:
                    ax.annotate(label, (lon, lat), xytext=(0, -15), textcoords='offset points', 
                                ha='center', va='top', fontsize=8, color='black',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'),
                                zorder=8)
                    labeled_dates.add(current_date)
    try:
        if os.path.exists(LOGO_PATH):
            logo_img = mpimg.imread(LOGO_PATH)
            # zoom 控制商標大小，請依據你的 png 原始解析度調整 (例如 0.1, 0.5 等)
            logo_box = OffsetImage(logo_img, zoom=0.03) 
            
            # xycoords='axes fraction' 代表使用圖表的相對比例 (0~1)
            # xy=(0.02, 0.98) 代表放在 X軸 2%、Y軸 98% 的位置 (左上角留一點邊距)
            # box_alignment=(0, 1) 代表以商標圖片的左上角為錨點進行對齊
            ab_logo = AnnotationBbox(logo_box, xy=(0.02, 0.98), xycoords='axes fraction', 
                                     box_alignment=(0, 1), frameon=False, zorder=10)
            ax.add_artist(ab_logo)
    except Exception as e:
        print(f"讀取商標失敗: {e}")
    # --- D. 標題與輸出 ---
    ax.set_title(f"中央氣象署預測路徑 ({title_time_str})", fontsize=18, pad=15, fontweight='bold')
    
    ax.outline_patch.set_visible(False) if hasattr(ax, 'outline_patch') else ax.spines['geo'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ 圖片已成功儲存至 {output_filename}")
    plt.close(fig)

if __name__ == "__main__":
    # 自動下載最新資料
    cwa_url = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com/Warning/W-C0034-005.json"
    download_cwa_json(cwa_url, JSON_PATH)
    
    # 產出一般版 (範圍：經度110-170，緯度5-35)
    print("\n正在繪製一般版地圖...")
    plot_typhoon_map(extent=[110, 170, 5, 35], output_filename="typhoon_path_map_normal.png")
    
    # 產出放大版 (範圍：經度115-130，緯度10-30)
    print("\n正在繪製放大版地圖...")
    plot_typhoon_map(extent=[115, 130, 15, 30], output_filename="typhoon_path_map_magnified.png")