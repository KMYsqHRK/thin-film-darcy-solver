import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.rcParams["font.family"] = "DejaVu Serif"   # 使用するフォント
plt.rcParams['axes.unicode_minus'] = False


# データ読み込み（パスは適宜変更してください）
df = pd.read_csv('../bin/output_pressure_grid.csv')

# プロットする時刻（5.0-6.5秒、0.3秒毎）
times_to_plot = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]

# 3行2列のサブプロット作成
fig = plt.figure(figsize=(8.3, 10))

# 全データから圧力範囲を取得（Z軸を統一するため）
z_min = df[df['WetDry'] == 1]['Pressure'].min()
z_max = df[df['WetDry'] == 1]['Pressure'].max()

for idx, time in enumerate(times_to_plot):
    # 指定時刻のデータを抽出
    df_time = df[df['Time'] == time].copy()
    
    if len(df_time) == 0:
        print(f"警告: 時刻 {time} のデータが見つかりません")
        continue
    
    # WetDry=0（乾燥）の場所は圧力を0に
    df_time.loc[df_time['WetDry'] == 0, 'Pressure'] = 0
    
    # グリッドデータを2次元配列に整形
    x_unique = sorted(df_time['x'].unique())
    y_unique = sorted(df_time['y'].unique())
    
    X, Y = np.meshgrid(x_unique, y_unique)
    Z = np.zeros_like(X)
    
    for i, y_val in enumerate(y_unique):
        for j, x_val in enumerate(x_unique):
            pressure_val = df_time[(df_time['x'] == x_val) & 
                                   (df_time['y'] == y_val)]['Pressure'].values
            if len(pressure_val) > 0:
                Z[i, j] = pressure_val[0]
            else:
                Z[i, j] = np.nan
    
    # サブプロット作成（3行2列）
    ax = fig.add_subplot(3, 2, idx + 1, projection='3d')
    
    # 3次元サーフェスプロット
    surf = ax.plot_surface(X, Y, Z, cmap=cm.bwr, 
                          edgecolor='none', alpha=0.8,
                          vmin=0, vmax=z_max)
    
    
    # 軸ラベル
    ax.set_xlabel('X [m]', fontsize=10)
    ax.set_ylabel('Y [m]', fontsize=10)
    ax.set_zlabel('Pressure [Pa]', fontsize=10)
    ax.set_title(f'Time = {time:.1f} s', fontsize=10, fontweight='bold', y=0.95)
    
    # Z軸の範囲を統一
    ax.set_zlim(z_min * 0.95, z_max * 1.05) 
    
    # 視点角度の設定
    ax.view_init(elev=25, azim=45)

#fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
plt.tight_layout()
plt.savefig('pressure_distribution_3d.png', dpi=300, bbox_inches='tight')
print("図を保存しました: pressure_distribution_3d.png")

plt.show()