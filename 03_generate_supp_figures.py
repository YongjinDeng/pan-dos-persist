# -*- coding: utf-8 -*-
r"""
================================================================================
03_GENERATE_SUPP_FIGURES.PY (BSPC SUPPLEMENTARY FIGURES S1 - S3)
================================================================================
生成 BSPC 补充材料 Figure S1 - S3。
注意：Figure S4 已移除。原脚本使用 np.random 伪造表面配准误差数据，
      属于学术不端。如需 S4，请先提供真实 HD95 测量值。

依赖前置条件：
    必须先运行 02_run_master_evaluation.py（已包含中间 CSV 导出补丁），
    生成以下文件：
        - Figure2_Longitudinal_Scatter_Data.csv
        - Figure3_ROC_Coordinates.csv
================================================================================
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
OUT_DIR = os.path.join(RESULT_DIR, "Route1_Master_Results")
os.makedirs(OUT_DIR, exist_ok=True)

SCATTER_CSV = os.path.join(OUT_DIR, "Figure2_Longitudinal_Scatter_Data.csv")
ROC_CSV = os.path.join(OUT_DIR, "Figure3_ROC_Coordinates.csv")

# 阈值常量（与 02_run_master_evaluation.py 保持一致）
T_V33 = 60.265
T_V40 = 24.716


def check_dependencies():
    """检查中间 CSV 是否已生成。"""
    missing = []
    if not os.path.exists(SCATTER_CSV):
        missing.append(SCATTER_CSV)
    if not os.path.exists(ROC_CSV):
        missing.append(ROC_CSV)
    if missing:
        raise FileNotFoundError(
            "缺少中间 CSV 文件。请先运行修复后的 02_run_master_evaluation.py，"
            "确保其中包含 df_long 和 roc_records 的 CSV 导出语句。\n"
            "缺失文件:\n  " + "\n  ".join(missing)
        )


# =============================================================================
# Figure S1: Bivariate Scatter with 95% Confidence Ellipses
# =============================================================================
def generate_figure_s1():
    df_scat = pd.read_csv(SCATTER_CSV)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    for ax, k1, k2, name, t_val in [
        (ax1, "CBCT1_V33", "CBCT2_V33", "V33", T_V33),
        (ax2, "CBCT1_V40", "CBCT2_V40", "V40", T_V40),
    ]:
        x = df_scat[k1].values
        y = df_scat[k2].values
        ax.scatter(x, y, c='#2980B9', edgecolors='k', s=60, alpha=0.85, zorder=3)

        lim = max(x.max(), y.max()) * 1.08
        ax.plot([0, lim], [0, lim], 'k--', lw=1.5, label='y = x', zorder=2)

        # 95% 置信椭圆
        cov = np.cov(x, y)
        mean = [np.mean(x), np.mean(y)]
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 1.96 * np.sqrt(vals)
        ell = Ellipse(
            xy=mean, width=width, height=height, angle=theta,
            edgecolor='#E74C3C', facecolor='none', lw=1.8, ls='--',
            label='95% confidence ellipse', zorder=4
        )
        ax.add_patch(ell)

        # 队列阈值参考线
        ax.axvline(t_val, color='#E74C3C', linestyle=':', alpha=0.6, lw=1.2)
        ax.axhline(t_val, color='#E74C3C', linestyle=':', alpha=0.6, lw=1.2)

        r, _ = stats.pearsonr(x, y)
        ax.set_xlabel(f"CBCT 1 {name} (cc)", fontweight='bold', fontsize=11)
        ax.set_ylabel(f"CBCT 2 {name} (cc)", fontweight='bold', fontsize=11)
        ax.set_title(f"Longitudinal {name} Dispersion (n = {len(x)}, r = {r:.3f})",
                     fontweight='bold', fontsize=12)
        ax.set_xlim(-lim * 0.05, lim)
        ax.set_ylim(-lim * 0.05, lim)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, loc='upper left', fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Figure_S1.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ Figure S1 已保存: {out}")


# =============================================================================
# Figure S2: Stratified Delta Dose Boxplots
# =============================================================================
def generate_figure_s2():
    df_scat = pd.read_csv(SCATTER_CSV)
    df_scat['Delta_V33'] = df_scat['CBCT2_V33'] - df_scat['CBCT1_V33']
    df_scat['Delta_V40'] = df_scat['CBCT2_V40'] - df_scat['CBCT1_V40']
    df_scat['CBCT1_V33_State'] = (df_scat['CBCT1_V33'] >= T_V33).astype(int)
    df_scat['CBCT1_V40_State'] = (df_scat['CBCT1_V40'] >= T_V40).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    for ax, d_col, s_col, name in [
        (ax1, 'Delta_V33', 'CBCT1_V33_State', 'V33'),
        (ax2, 'Delta_V40', 'CBCT1_V40_State', 'V40'),
    ]:
        g0 = df_scat[df_scat[s_col] == 0][d_col].values
        g1 = df_scat[df_scat[s_col] == 1][d_col].values

        bp = ax.boxplot(
            [g0, g1],
            patch_artist=True, widths=0.45,
            medianprops=dict(color='black', linewidth=1.5)
        )
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Low-Risk (n={len(g0)})', f'High-Risk (n={len(g1)})'])

        bp['boxes'][0].set_facecolor('#4D8CB5')
        bp['boxes'][0].set_alpha(0.75)
        bp['boxes'][1].set_facecolor('#D55E00')
        bp['boxes'][1].set_alpha(0.75)

        ax.axhline(0, color='r', linestyle='--', alpha=0.7, lw=1.2)
        ax.set_ylabel(f"Δ{name} (CBCT2 - CBCT1, cc)", fontweight='bold', fontsize=11)
        ax.set_title(f"Interfractional Δ{name} by Early Risk State",
                     fontweight='bold', fontsize=12)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Figure_S2.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ Figure S2 已保存: {out}")


# =============================================================================
# Figure S3: Full-Resolution ROC Curves
# =============================================================================
def generate_figure_s3():
    df_roc = pd.read_csv(ROC_CSV)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    colors = {
        "M0: Planning Physical DVH": "#7F8C8D",
        "M0b: CBCT1 Physical DVH": "#E74C3C",
        "M1: CBCT1 Whole-OAR Dosiomics": "#2980B9",
        "M2: Dose-Gradient Dosiomics": "#27AE60",
        "M3: Gradient + Uncertainty Dosiomics": "#8E44AD",
    }

    for m_name, grp in df_roc.groupby("Model"):
        c = colors.get(m_name, 'black')
        grp_sorted = grp.sort_values('FPR')
        ax.plot(grp_sorted['FPR'], grp_sorted['TPR'], color=c, lw=2.2,
                label=m_name.split(':')[0])

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontweight='bold', fontsize=11)
    ax.set_ylabel("True Positive Rate", fontweight='bold', fontsize=11)
    ax.set_title("Full-Resolution Patient-Level ROC (M0 - M3)",
                 fontweight='bold', fontsize=12)
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "Figure_S3.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ Figure S3 已保存: {out}")


# =============================================================================
# 主入口
# =============================================================================
def main():
    print("=" * 72)
    print("BSPC SUPPLEMENTARY FIGURES GENERATOR (S1 - S3)")
    print(f"Output Directory: {OUT_DIR}")
    print("=" * 72)

    check_dependencies()

    print("\n>>> 生成 Figure S1 ...")
    generate_figure_s1()

    print(">>> 生成 Figure S2 ...")
    generate_figure_s2()

    print(">>> 生成 Figure S3 ...")
    generate_figure_s3()

    print("\n" + "=" * 72)
    print("✅ Figure S1 - S3 全部生成成功！")
    print("⚠️  Figure S4 已移除：原脚本使用 np.random 伪造表面配准误差数据。")
    print("     如需 S4，请先提供真实 HD95 测量值。")
    print("=" * 72)


if __name__ == "__main__":
    main()