#!/usr/bin/env python3
"""
Himawari-9 衛星雲圖繪製工具
支援波段: Band 3 (IR), Band 6 (WV), Band 8 (VIS+IR), Band 9 (VIS 1km)
色階: ir, rammb, ca, rbtop, davork, wv, vis
"""

# =============================================================================
# ★ 使用者設定區（請在此填寫路徑）★
# =============================================================================

INPUT_DIR  = "./"   # 資料夾路徑（含 BASE-IMG*.btfile.txt 與 xyinfo-*.txt）
OUTPUT_DIR = "./"  # 輸出圖片資料夾路徑

# =============================================================================

import os
import re
import glob
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import json
import urllib3
# 忽略 urllib3 的不安全請求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# 色階定義
# =============================================================================

def make_ir_cmap():
    """IR 色階 (K → °C)，使用 BoundaryNorm 控制"""
    # 原始定義在 K，轉成°C: T_C = T_K - 273.15
    points = [
        [150 - 273.15, [0, 0, 0]],
        [165 - 273.15, [94, 94, 94]],
        [185 - 273.15, [255, 255, 255]],
        [186 - 273.15, [0, 0, 0]],
        [200 - 273.15, [255, 0, 0]],
        [215 - 273.15, [255, 255, 0]],
        [225 - 273.15, [0, 255, 0]],
        [230 - 273.15, [50, 50, 140]],
        [240 - 273.15, [17, 192, 233]],
        [245 - 273.15, [52, 229, 248]],
        [251 - 273.15, [255, 255, 255]],
        [320 - 273.15, [0, 0, 0]]
    ]
    vmin = 150 - 273.15
    vmax = 320 - 273.15
    colors_rgb = [[r/255, g/255, b/255] for _, (r, g, b) in points]
    positions = [(t - vmin) / (vmax - vmin) for t, _ in points]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, pos in enumerate(positions):
        cdict['red'].append((pos, colors_rgb[i][0], colors_rgb[i][0]))
        cdict['green'].append((pos, colors_rgb[i][1], colors_rgb[i][1]))
        cdict['blue'].append((pos, colors_rgb[i][2], colors_rgb[i][2]))
    cmap = LinearSegmentedColormap('ir', cdict, N=2048)
    return cmap, vmin, vmax


def make_rammb_cmap():
    cdict = {
        'red': [(0.0, 0.0, 0.3137254901960784),
                (0.06666666666666667, 1.0, 0.3137254901960784),
                (0.13333333333333333, 1.0, 1.0),
                (0.2, 0.39215686274509803, 0.0),
                (0.26666666666666666, 0.0, 0.0),
                (0.3333333333333333, 0.0, 0.3333333333333333),
                (0.4666666666666667, 0.7058823529411765, 1.0),
                (0.9, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'blue': [(0.0, 0.0, 0.3137254901960784),
                 (0.06666666666666667, 1.0, 0.3137254901960784),
                 (0.13333333333333333, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.26666666666666666, 0.0, 1.0),
                 (0.3333333333333333, 0.39215686274509803, 0.3333333333333333),
                 (0.4666666666666667, 1.0, 1.0),
                 (0.9, 0.0, 0.0),
                 (1.0, 0.0, 0.0)],
        'green': [(0.0, 0.0, 0.3137254901960784),
                  (0.06666666666666667, 1.0, 0.3137254901960784),
                  (0.13333333333333333, 1.0, 0.0),
                  (0.2, 0.0, 1.0),
                  (0.26666666666666666, 0.39215686274509803, 0.0),
                  (0.3333333333333333, 0.0, 0.3333333333333333),
                  (0.4666666666666667, 1.0, 1.0),
                  (0.9, 0.0, 0.0),
                  (1.0, 0.0, 0.0)]
    }
    cmap = LinearSegmentedColormap('rammb', cdict, N=2048)
    return cmap, 50.0, -100.0


def make_ca_cmap():
    cdict = {
        'green': [(0.0, 0.0, 1.0), (0.06666666666666667, 0.7843137254901961, 0.7843137254901961), (0.08666666666666667, 0.5490196078431373, 0.5490196078431373),
                  (0.1, 0.21176470588235294, 0.21176470588235294), (0.12666666666666668, 0.40784313725490196, 0.40784313725490196), (0.14, 0.2823529411764706, 0.2823529411764706),
                  (0.16, 0.1568627450980392, 0.1568627450980392), (0.17333333333333334, 0.09411764705882353, 0.09411764705882353), (0.2, 0.25098039215686274, 0.25098039215686274),
                  (0.24, 0.6901960784313725, 0.6901960784313725), (0.2733333333333333, 0.9254901960784314, 0.9254901960784314), (0.30666666666666664, 0.9882352941176471, 0.9882352941176471),
                  (0.38666666666666666, 0.7686274509803922, 0.7686274509803922), (0.46, 0.5176470588235295, 0.5176470588235295), (0.5733333333333334, 0.2980392156862745, 0.2980392156862745),
                  (0.6666666666666666, 0.22745098039215686, 0.22745098039215686), (0.7066666666666667, 0.20392156862745098, 0.20392156862745098), (0.72, 0.3215686274509804, 0.3215686274509804),
                  (0.7266666666666667, 0.40784313725490196, 0.40784313725490196), (0.7666666666666667, 0.25098039215686274, 0.25098039215686274), (0.8, 0.12549019607843137, 0.12549019607843137),
                  (0.8333333333333334, 0.047058823529411764, 0.047058823529411764), (0.8666666666666667, 0.0, 0.0), (0.9333333333333333, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'red': [(0.0, 0.0, 1.0), (0.06666666666666667, 0.7843137254901961, 0.7843137254901961), (0.08666666666666667, 0.5490196078431373, 0.5490196078431373),
                (0.1, 0.3137254901960784, 0.3137254901960784), (0.126666666666666168, 0.6588235294117647, 0.6588235294117647), (0.14, 0.7372549019607844, 0.7372549019607844),
                (0.16, 0.7686274509803922, 0.7686274509803922), (0.17333333333333334, 0.8784313725490196, 0.8784313725490196), (0.2, 0.9882352941176471, 0.9882352941176471),
                (0.24, 0.9725490196078431, 0.9725490196078431), (0.2733333333333333, 0.9254901960784314, 0.9254901960784314), (0.30666666666666664, 0.48627450980392156, 0.48627450980392156),
                (0.38666666666666666, 0.09411764705882353, 0.09411764705882353), (0.46, 0.03137254901960784, 0.03137254901960784), (0.5733333333333334, 0.09411764705882353, 0.09411764705882353),
                (0.6666666666666666, 0.047058823529411764, 0.047058823529411764), (0.7066666666666667, 0.06274509803921569, 0.06274509803921569), (0.72, 0.26666666666666666, 0.26666666666666666),
                (0.7266666666666667, 0.40784313725490196, 0.40784313725490196), (0.7666666666666667, 0.25098039215686274, 0.25098039215686274), (0.8, 0.12549019607843137, 0.12549019607843137),
                (0.8333333333333334, 0.047058823529411764, 0.047058823529411764), (0.8666666666666667, 0.0, 0.0), (0.9333333333333333, 0.0, 0.0), (1.0, 0.5176470588235295, 0.0)],
        'blue': [(0.0, 0.0, 1.0), (0.06666666666666667, 1.0, 1.0), (0.08666666666666667, 0.9411764705882353, 0.9411764705882353), (0.1, 0.6901960784313725, 0.6901960784313725),
                 (0.12666666666666668, 0.9098039215686274, 0.9098039215686274), (0.14, 0.4235294117647059, 0.4235294117647059), (0.16, 0.1411764705882353, 0.1411764705882353),
                 (0.17333333333333334, 0.047058823529411764, 0.047058823529411764), (0.2, 0.0, 0.0), (0.24, 0.0, 0.0), (0.2733333333333333, 0.0, 0.0), (0.30666666666666664, 0.0, 0.0),
                 (0.38666666666666666, 0.4235294117647059, 0.4235294117647059), (0.46, 0.5803921568627451, 0.5803921568627451), (0.5733333333333334, 0.4549019607843137, 0.4549019607843137),
                 (0.6666666666666666, 0.32941176470588235, 0.32941176470588235), (0.7066666666666667, 0.2980392156862745, 0.2980392156862745), (0.72, 0.3607843137254902, 0.3607843137254902),
                 (0.7266666666666667, 0.40784313725490196, 0.40784313725490196), (0.7666666666666667, 0.25098039215686274, 0.25098039215686274), (0.8, 0.12549019607843137, 0.12549019607843137),
                 (0.8333333333333334, 0.047058823529411764, 0.047058823529411764), (0.8666666666666667, 0.0, 0.0), (0.9333333333333333, 0.0, 0.0), (1.0, 0.0, 0.0)]
    }
    cmap = LinearSegmentedColormap('ca', cdict, N=2048)
    return cmap, 50.0, -100.0


def make_rbtop_cmap():
    cdict = {
        'red': [(0.0, 0.0, 1.0), (0.190476, 0.0, 0.0), (0.269841, 1.0, 1.0), (0.349206, 1.0, 1.0), (0.428571, 0.0, 0.0),
                (0.555556, 0.0, 0.0), (0.658730, 0.980392, 0.980392), (0.84, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'green': [(0.0, 0.0, 1.0), (0.190476, 0.0, 0.0), (0.269841, 0.0, 0.0), (0.349206, 1.0, 1.0), (0.428571, 1.0, 1.0),
                  (0.555556, 0.0, 0.0), (0.619048, 0.0, 0.0), (0.658730, 0.980392, 0.980392), (0.84, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'blue': [(0.0, 0.0, 1.0), (0.190476, 0.0, 0.0), (0.428571, 0.0, 0.0), (0.555556, 1.0, 1.0), (0.619048, 1.0, 1.0),
                 (0.658730, 0.980392, 0.980392), (0.84, 0.0, 0.0), (1.0, 0.0, 0.0)]
    }
    cmap = LinearSegmentedColormap('rbtop', cdict, N=2048)
    return cmap, 50.0, -100.0


def make_davork_cmap():
    """davork 色階 + 暖區平滑漸層"""
    vmin, vmax = -100.0, 50.0
    N = 4096
    colors_array = np.zeros((N, 4))
    colors_array[:, 3] = 1.0  # 設定 Alpha 不透明度為 1

    def t_to_idx(t):
        idx = int(round((t - vmin) / (vmax - vmin) * (N - 1)))
        return max(0, min(N - 1, idx))

    def hex_to_rgb(hx):
        hx = hx.lstrip('#')
        return tuple(int(hx[i:i+2], 16)/255.0 for i in (0, 2, 4))

    def fill_solid(t_lo, t_hi, rgb):
        i0 = t_to_idx(t_lo)
        i1 = t_to_idx(t_hi)
        colors_array[i0:i1+1, 0] = rgb[0]
        colors_array[i0:i1+1, 1] = rgb[1]
        colors_array[i0:i1+1, 2] = rgb[2]

    def fill_gradient(t_lo, t_hi, hex_lo, hex_hi):
        i0 = t_to_idx(t_lo)
        i1 = t_to_idx(t_hi)
        if i0 == i1: return
        r0, g0, b0 = hex_to_rgb(hex_lo)
        r1, g1, b1 = hex_to_rgb(hex_hi)
        length = i1 - i0 + 1
        colors_array[i0:i1+1, 0] = np.linspace(r0, r1, length)
        colors_array[i0:i1+1, 1] = np.linspace(g0, g1, length)
        colors_array[i0:i1+1, 2] = np.linspace(b0, b1, length)

    # 1. 填入深對流冷雲頂區 (純色塊)
    fill_solid(-100, -80, [0x58/255, 0x58/255, 0x58/255]) # 深灰
    fill_solid( -80, -75, [0xff/255, 0x00/255, 0x00/255]) # 紅
    fill_solid( -75, -69, [0x66/255, 0xff/255, 0xff/255]) # 青
    fill_solid( -69, -63, [0xff/255, 0x00/255, 0xff/255]) # 洋紅
    fill_solid( -63, -53, [0xff/255, 0xff/255, 0x00/255]) # 黃
    fill_solid( -53, -41, [0x00/255, 0xff/255, 0x00/255]) # 綠
    fill_solid( -41, -30, [0x00/255, 0x00/255, 0xff/255]) # 藍

    # 2. 填入新增的暖區漸層 (自動線性插值)
    fill_gradient(-30, 9, '#d2d2d2', '#3a3a3a')
    fill_gradient(9,  50, '#fafafa', '#292929')

    cmap = ListedColormap(colors_array, name='davork')
    return cmap, vmin, vmax


def make_wv_cmap():
    cdict = {
        'green': [(0, 0/255, 0/255),
                  (30/150, 0/255, 0/255),
                  (45.5/150, 128/255, 128/255),
                  (58.5/150, 255/255, 255/255),
                  (65/150, 255/255, 255/255),
                  (72.5/150, 255/255, 255/255),
                  (86/150, 128/255, 128/255),
                  (100/150, 20/255, 255/255),
                  (1, 1, 1)],
        'red':   [(0, 128/255, 128/255),
                  (30/150, 128/255, 128/255),
                  (45.5/150, 255/255, 255/255),
                  (58.5/150, 255/255, 255/255),
                  (65/150, 128/255, 128/255),
                  (72.5/150, 128/255, 128/255),
                  (86/150, 0/255, 0/255),
                  (100/150, 100/255, 255/255),
                  (1, 1, 1)],
        'blue':  [(0, 0/255, 0/255),
                  (30/150, 0/255, 0/255),
                  (45.5/150, 0/255, 0/255),
                  (58.5/150, 128/255, 128/255),
                  (65/150, 128/255, 128/255),
                  (72.5/150, 255/255, 255/255),
                  (86/150, 255/255, 255/255),
                  (100/150, 100/255, 255/255),
                  (1, 1, 1)]
    }
    cmap = LinearSegmentedColormap('wv', cdict, N=2048)
    return cmap, 50.0, -100.0


def make_vis_cmap():
    """可見光: 黑→白 (高溫到低溫)"""
    cmap = plt.cm.gray_r
    return cmap, None, None  # vmin/vmax 從資料本身決定


# =============================================================================
# 色階查找表
# =============================================================================
CMAP_FACTORY = {
    'ir':     make_ir_cmap,
    'rammb':  make_rammb_cmap,
    'ca':     make_ca_cmap,
    'rbtop':  make_rbtop_cmap,
    'davork': make_davork_cmap,
    'wv':     make_wv_cmap,
    'vis':    make_vis_cmap,
}

# 波段 → 預設色階
BAND_DEFAULT_CMAP = {
    3: 'ir',
    6: 'wv',
    8: 'vis+ir',
}

# 波段 → 單位說明
BAND_UNIT_LABEL = {
    3: '°C',
    6: '°C',
    8: '°C',
}

# =============================================================================
# 檔名解析
# =============================================================================

def parse_btfile_name(filename):
    """
    解析 BASE-IMG3-8.04W.btfile.txt 格式的檔名
    回傳: (band_type, resolution_km, sat_id)
    band_type: 3/6/8/9
    resolution_km: 1/2/4/8
    sat_id: e.g. '04W'
    """
    basename = os.path.basename(filename)
    # 格式: BASE-IMG{band}-{res}.{sat_id}.btfile.txt
    m = re.match(r'BASE-IMG(\d+)-(\d+)\.(\d+[A-Z]+)\.btfile\.txt', basename, re.IGNORECASE)
    if not m:
        raise ValueError(f"無法解析檔名: {basename}")
    band_type = int(m.group(1))
    resolution_km = int(m.group(2))
    sat_id = m.group(3)
    return band_type, resolution_km, sat_id


def find_xyinfo(data_dir, resolution_km, sat_id):
    """依 resolution_km 與 sat_id 找對應的 xyinfo 檔"""
    fname = f"xyinfo-{resolution_km}.{sat_id}.txt"
    full_path = os.path.join(data_dir, fname)
    if os.path.exists(full_path):
        return full_path
    # 也嘗試在同層目錄
    candidates = glob.glob(os.path.join(data_dir, f"xyinfo-{resolution_km}*.txt"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"找不到 xyinfo 檔: {fname} (在 {data_dir})")


# =============================================================================
# 資料讀取 (完全照抄原始邏輯)
# =============================================================================

def read_btfile(path):
    data = np.loadtxt(path)
    return data


def try_reshape(data):
    size = data.size
    # ★ 修正重點：將 (800, 600) 移至第一順位，避免 48 萬像素被錯誤折疊
    candidates = [
        (800, 600),
        (600, 800),
        (881, 921),
        (921, 881),
        (720, 1440),
    ]
    for r, c in candidates:
        if r * c == size:
            print(f"  ✔ reshape: {r} x {c}")
            return data.reshape(r, c)
            
    # 若都不符合，嘗試找整數因子對
    sqrt_size = int(np.sqrt(size))
    for r in range(sqrt_size, 0, -1):
        if size % r == 0:
            c = size // r
            print(f"  ✔ reshape (auto): {r} x {c}")
            return data.reshape(r, c)
            
    raise ValueError(f"無法判斷 grid，資料點數={size}")

def make_vis_cmap():
    """可見光: 黑→白 (低反照率到高反照率)"""
    # ★ 修正重點：將 gray_r 改為 gray，確保雲層是白色的
    cmap = plt.cm.gray
    return cmap, None, None  # vmin/vmax 從資料本身決定


def read_xyinfo(path):
    """完全照抄原始 plot.py 邏輯"""
    with open(path, "r") as f:
        line = f.readline().strip()
    lat_min, lon_min, dlat, dlon = map(float, line.split("|"))
    return lat_min, lon_min, dlat, dlon


def build_latlon(grid, lat_min, lon_min, dlat, dlon):
    """建立經緯度網格（修正經度起點偏移問題）"""
    ny, nx = grid.shape

    # 緯度（左上角 → 往下遞減）
    lats = lat_min - np.arange(ny) * abs(dlat)

    # 經度：原始負號代表東經 → 先取絕對值
    lon_start = abs(lon_min)
    # 使用 arange 確保從起點正確計算，避免 cumsum 造成的 1 像素偏移
    lons = lon_start + np.arange(nx) * abs(dlon)

    return lats, lons


def plot_satellite(btfile_path, xyinfo_path, output_path, cmap_name, band_type, resolution_km):
    # ★ 修正重點 1: 每次繪圖前先關閉所有未釋放的圖片
    plt.close('all')
    
    print(f"  讀取資料: {btfile_path}")
    data_raw = read_btfile(btfile_path)
    grid = try_reshape(data_raw) if data_raw.ndim == 1 else data_raw
    ny, nx = grid.shape
    print(f"  grid shape: {ny} x {nx}")

    print(f"  讀取 xyinfo: {xyinfo_path}")
    lat_min, lon_min, dlat, dlon = read_xyinfo(xyinfo_path)

    lats, lons = build_latlon(grid, lat_min, lon_min, dlat, dlon)
    plot_data = grid.copy()

    if cmap_name not in CMAP_FACTORY:
        raise ValueError(f"未知色階: {cmap_name}")

    cmap, vmin, vmax = CMAP_FACTORY[cmap_name]()

    # =========================
    # VIS / BT contrast fix
    # =========================
    if vmin is None:
        valid = plot_data[plot_data > -100]

        if valid.size > 0:
            vmin = float(np.nanpercentile(valid, 5))
            vmax = float(np.nanpercentile(valid, 95))

            if abs(vmax - vmin) < 1e-6:
                vmin = float(np.min(valid))
                vmax = float(np.max(valid))
        else:
            vmin, vmax = 0.0, 1.0

    # mask invalid
    masked_data = np.ma.masked_where(plot_data <= -100.0, plot_data)

# ★ 修正重點 2: 使用預設投影，避免 Shapely 2.0+ 與 Cartopy gridlines 發生邊界不閉合的錯誤
    data_proj = ccrs.PlateCarree()
    plot_proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_subplot(1, 1, 1, projection=plot_proj)

    # extent
    lon_ext = [lons[0], lons[-1]]
    lat_ext = [lats[-1], lats[0]]
    ax.set_extent([lon_ext[0], lon_ext[1], lat_ext[0], lat_ext[1]], crs=data_proj)

    # =========================
    # 白底地圖底層
    # =========================
    ax.add_feature(cfeature.LAND.with_scale('10m'),
                   facecolor='#f2f2f2', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'),
                   facecolor='#ffffff', zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'),
                   linewidth=0.5, edgecolor='#888888', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                   linewidth=0.8, edgecolor='#444444', zorder=2)

    # =========================
    # image layer (★ 改用 pcolormesh 避開 scipy 依賴)
    # =========================
    im = ax.pcolormesh(
        lons, lats, masked_data,
        transform=data_proj,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto',
        zorder=1
    )

    # =========================
    # gridlines (white theme)
    # =========================
    gl = ax.gridlines(
        crs=data_proj,
        draw_labels=True,
        linewidth=0.5,
        color='#cccccc',
        alpha=0.8,
        linestyle='--',
        zorder=4
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 9, 'color': '#333333'}
    gl.ylabel_style = {'size': 9, 'color': '#333333'}

    # =========================
    # colorbar (white theme)
    # =========================
    cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                        fraction=0.025, pad=0.03, shrink=0.7)
    cbar.ax.tick_params(labelsize=9, colors='#333333')
    cbar.outline.set_edgecolor('#bbbbbb')

    unit_label = BAND_UNIT_LABEL.get(band_type, '°C')

    if band_type in [8, 9]:
        cbar.set_label('Albedo', fontsize=10, color='#333333')
        title = f"Himawari-9 Band 03 Albedo ({cmap_name}_{resolution_km}km)"
    elif band_type in [6]:
        cbar.set_label(f'Brightness Temperature ({unit_label})',
                       fontsize=10, color='#333333')
        title = f"Himawari-9 Band 08 BT ({cmap_name}_{resolution_km}km)"
    else:
        cbar.set_label(f'Brightness Temperature ({unit_label})',
                       fontsize=10, color='#333333')
        title = f"Himawari-9 Band 13 BT ({cmap_name}_{resolution_km}km)"

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12, color='#222222')

    # =========================
    # figure background (white theme)
    # =========================
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plt.tight_layout()

    # ★ 修正重點 3: 使用 try...finally 確保強制釋放 fig
    try:
        plt.savefig(
            output_path,
            dpi=120,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
    finally:
        plt.close(fig)

    print(f"  ✔ 已儲存: {output_path}")

# =============================================================================
# 波段 → 色階對應規則
# Band 3 (IR)  → 所有 IR 色階都產生：ir, rammb, ca, rbtop, davork
# Band 6 (WV)  → wv
# Band 8 (VIS+IR) → vis
# Band 9 (VIS 1km) → vis
# =============================================================================

IR_CMAPS  = ['ir', 'rammb', 'ca', 'rbtop', 'davork']   # Band 3 全部產出
WV_CMAPS  = ['wv']                                        # Band 6
VIS_CMAPS = ['vis']                                       # Band 8 / 9

BAND_CMAPS = {
    3: IR_CMAPS,
    6: WV_CMAPS,
    8: VIS_CMAPS,
    9: VIS_CMAPS,
}


def get_cmaps_for_band(band_type):
    """依波段回傳要產生的色階清單"""
    return BAND_CMAPS.get(band_type, ['ir'])


def process_directory(input_dir, output_dir):
    """
    掃描資料夾內所有 BASE-IMG*.btfile.txt 並繪圖。
    Band 3 自動產出所有 IR 色階；其他波段使用對應色階。
    """
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(input_dir, "BASE-IMG*.btfile.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[警告] 在 {input_dir} 中找不到任何 BASE-IMG*.btfile.txt")
        return

    print(f"找到 {len(files)} 個資料檔案")

    total_ok = 0
    total_err = 0

    for btfile in files:
        print(f"\n{'─'*55}")
        print(f"處理: {os.path.basename(btfile)}")
        try:
            band_type, resolution_km, sat_id = parse_btfile_name(btfile)
            print(f"  波段={band_type}, 解析度={resolution_km}km, 衛星ID={sat_id}")

            xyinfo_path = find_xyinfo(input_dir, resolution_km, sat_id)

            cmap_list = get_cmaps_for_band(band_type)
            print(f"  色階清單: {cmap_list}")

            for cmap_name in cmap_list:
                # 輸出檔名: Himawari9_B{band}_{cmap}_{sat_id}.png
                out_name = f"Himawari9_B{band_type:02d}_{cmap_name}_{resolution_km}km.png"
                out_path = os.path.join(output_dir, out_name)
                print(f"  → [{cmap_name}]", end=" ", flush=True)
                try:
                    plot_satellite(btfile, xyinfo_path, out_path, cmap_name, band_type,resolution_km)
                    total_ok += 1
                except Exception as e:
                    print(f"\n    [錯誤] {e}")
                    import traceback
                    traceback.print_exc()
                    total_err += 1

        except Exception as e:
            print(f"  [錯誤] 無法處理檔案: {e}")
            import traceback
            traceback.print_exc()
            total_err += 1

    print(f"\n{'='*55}")
    print(f"完成！成功={total_ok} 張，錯誤={total_err} 張")
    print(f"輸出資料夾: {output_dir}")

# =============================================================================
# 自動下載功能
# =============================================================================
# =============================================================================
# 自動下載功能 (修改版：改用 TXT 讀取颱風編號)
# =============================================================================
def download_typhoon_data(input_dir):
    print("\n" + "="*55)
    print("  開始自動下載最新颱風資料")
    print("="*55)

    # 1. 清空資料夾內的 txt 檔案
    old_txts = glob.glob(os.path.join(input_dir, "*.txt"))
    for f in old_txts:
        try:
            os.remove(f)
        except Exception as e:
            print(f"  [警告] 無法刪除舊檔案 {f}: {e}")
    if old_txts:
        print(f"  ✔ 已清空 {len(old_txts)} 個舊檔案")

    # 2. 抓取 TXT 尋找颱風 ID
    http = urllib3.PoolManager(cert_reqs='CERT_NONE')
    txt_url = "https://seanthinkweather.dpdns.org/tynumsat.txt"
    print(f"  正在獲取風暴清單: {txt_url}")
    
    try:
        r = http.request('GET', txt_url, timeout=10.0)
        if r.status != 200:
            print(f"  [錯誤] TXT 取得失敗，HTTP 狀態碼: {r.status}")
            return
        
        # 直接讀取並解碼為文字
        text_data = r.data.decode('utf-8').strip()
    except Exception as e:
        print(f"  [錯誤] 無法取得或解析 TXT: {e}")
        return

    # 利用正則表達式找出所有符合 ??W 格式的字串 (例如 04W)，並使用 set 去除重複項目
    storm_ids = list(set(re.findall(r'\d{2}W', text_data)))
    
    if not storm_ids:
        print("  [提示] 目前文字檔中沒有活躍的西北太平洋颱風 (??W)")
        return
        
    print(f"  發現目標颱風 ID: {', '.join(storm_ids)}")

    # 3. 執行下載
    base_url = "https://tropic.ssec.wisc.edu/real-time/westpac/storm/images/"
    bands = [3, 6, 8]
    resolutions = [2, 4, 8]

    for sid in storm_ids:
        print(f"\n  ➤ 開始下載颱風 {sid} 的圖資...")
        for res in resolutions:
            # 抓取 xyinfo
            xy_name = f"xyinfo-{res}.{sid}.txt"
            xy_url = base_url + xy_name
            try:
                r = http.request('GET', xy_url, timeout=15.0)
                if r.status == 200:
                    with open(os.path.join(input_dir, xy_name), 'wb') as f:
                        f.write(r.data)
                    print(f"    ✔ 成功: {xy_name}")
                else:
                    print(f"    ✘ 找不到: {xy_name}")
            except Exception as e:
                print(f"    ✘ 失敗: {xy_name} ({e})")

            # 抓取 波段資料
            for band in bands:
                bt_name = f"BASE-IMG{band}-{res}.{sid}.btfile.txt"
                bt_url = base_url + bt_name
                try:
                    r = http.request('GET', bt_url, timeout=30.0)
                    if r.status == 200:
                        with open(os.path.join(input_dir, bt_name), 'wb') as f:
                            f.write(r.data)
                        print(f"    ✔ 成功: {bt_name}")
                    else:
                        print(f"    ✘ 找不到: {bt_name}")
                except Exception as e:
                    print(f"    ✘ 失敗: {bt_name} ({e})")

    print("  ✔ 所有下載任務結束。")

# =============================================================================
# 主程式入口
# =============================================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  Himawari-9 颱風自動追蹤與繪製工具")
    print("=" * 55)
    
    # 確保資料夾存在
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 執行自動下載
    download_typhoon_data(INPUT_DIR)
    
    # 2. 執行繪圖
    print("\n" + "="*55)
    print("  開始處理影像資料")
    print("="*55)
    process_directory(INPUT_DIR, OUTPUT_DIR)
