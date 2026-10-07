# -*- coding: utf-8 -*-
"""
================================================================================
02_RUN_MASTER_EVALUATION.PY (DEFINITIVE COMPLETE REVISION)
================================================================================
修复说明:
1. 修复 run_module_3 中 df_long 未定义的 NameError 作用域 Bug。
2. 将 df_long 规范化在各模块间传递。
3. 调整 M2/M3 的评估正则化，展示真实的高维组学退化数值。
4. 完整输出 Module 1 - 4 全部结果、主表 (Table 1-4) 与主图 (Figure 1-4)。
================================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import warnings

warnings.filterwarnings("ignore")

# ----------------- 路径与环境 -----------------
RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
OUT_DIR = os.path.join(RESULT_DIR, "Route1_Master_Results")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_FILE = os.path.join(RESULT_DIR, "Longitudinal_DoseGradient_Features_v2.csv")
LABEL_FILE = os.path.join(RESULT_DIR, "clinical_labels_detailed_v3.csv")

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
N_BOOTSTRAP = 2000
RANDOM_STATE = 42

# ----------------- 统计学工具函数 -----------------
def calc_pearson_ci(r, n, alpha=0.05):
    if not np.isfinite(r) or n <= 3 or abs(r) >= 1.0:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))

def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    boot_aucs, boot_briers = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        boot_briers.append(brier_score_loss(y_true[idx], y_prob[idx]))
    return (float(np.median(boot_aucs)), 
            (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))),
            float(np.median(boot_briers)),
            boot_aucs)

def evaluate_model_pipeline(X, y, penalty='l2', C=1.0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(
            penalty=penalty, 
            solver='liblinear' if penalty == 'l1' else 'lbfgs', 
            C=C, 
            class_weight='balanced', 
            random_state=RANDOM_STATE, 
            max_iter=5000
        )
        pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('clf', clf)])
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return oof

# =============================================================================
# 模块 1: 队列基准特征与终点定义 (TABLE 1)
# =============================================================================
def run_module_1(labels_df):
    print("\n" + "="*80)
    print(">>> [MODULE 1] 队列基线剂量学特征统计 (Table 1)")
    print("="*80)
    
    t_v33 = float(labels_df["V33_Gastroduodenal_cc"].median())
    t_v40 = float(labels_df["V40_Gastroduodenal_cc"].median())
    
    labels_df["Group"] = labels_df["Label"].map({0: "Low Risk (0)", 1: "High Risk (1)"})
    table1 = labels_df.groupby("Group").agg({
        "Organ_Volume_cc": ["mean", "std"],
        "V33_Gastroduodenal_cc": ["mean", "std"],
        "V40_Gastroduodenal_cc": ["mean", "std"],
        "Max_Dose_Gy": ["mean", "std"]
    }).round(2)
    
    table1_path = os.path.join(OUT_DIR, "Table1_Cohort_Characteristics.csv")
    table1.to_csv(table1_path)
    print(f"Table 1 已保存: {table1_path}")
    print(table1)
    return t_v33, t_v40

# =============================================================================
# 模块 2: 纵向连续持久性与状态转移 (TABLE 2, FIGURE 1 & 2)
# =============================================================================
def run_module_2(features_df, t_v33, t_v40):
    print("\n" + "="*80)
    print(">>> [MODULE 2] 纵向剂量持久性统计与状态转移分析 (Table 2 & Figures)")
    print("="*80)

    c1 = features_df[features_df["CBCT_ID"] == "CBCT_1"].set_index("Patient_ID")
    c2 = features_df[features_df["CBCT_ID"] == "CBCT_2"].set_index("Patient_ID")
    common_pids = sorted(list(set(c1.index) & set(c2.index)))
    
    df_long = pd.DataFrame({
        "Patient_ID": common_pids,
        "CBCT1_V33": c1.loc[common_pids, "V33_cc"].values,
        "CBCT2_V33": c2.loc[common_pids, "V33_cc"].values,
        "CBCT1_V40": c1.loc[common_pids, "V40_cc"].values,
        "CBCT2_V40": c2.loc[common_pids, "V40_cc"].values,
    })
    
    df_long["CBCT1_V33_State"] = (df_long["CBCT1_V33"] >= t_v33).astype(int)
    df_long["CBCT2_V33_State"] = (df_long["CBCT2_V33"] >= t_v33).astype(int)
    df_long["CBCT1_V40_State"] = (df_long["CBCT1_V40"] >= t_v40).astype(int)
    df_long["CBCT2_V40_State"] = (df_long["CBCT2_V40"] >= t_v40).astype(int)
    df_long["CBCT1_Combined"] = ((df_long["CBCT1_V33_State"] == 1) | (df_long["CBCT1_V40_State"] == 1)).astype(int)
    df_long["CBCT2_Combined"] = ((df_long["CBCT2_V33_State"] == 1) | (df_long["CBCT2_V40_State"] == 1)).astype(int)

    # 连续性分析
    continuity_rows = []
    for metric, k1, k2, thresh in [("V33", "CBCT1_V33", "CBCT2_V33", t_v33), ("V40", "CBCT1_V40", "CBCT2_V40", t_v40)]:
        x, y = df_long[k1].values, df_long[k2].values
        r, p_pearson = stats.pearsonr(x, y)
        rho, p_spearman = stats.spearmanr(x, y)
        r_lo, r_hi = calc_pearson_ci(r, len(x))
        delta = y - x
        continuity_rows.append({
            "Endpoint": metric,
            "N": len(x),
            "Threshold_cc": thresh,
            "Pearson_r": round(r, 4),
            "Pearson_95CI": f"{r:.4f} ({r_lo:.4f} - {r_hi:.4f})",
            "Spearman_rho": round(rho, 4),
            "Mean_Change_cc": round(float(np.mean(delta)), 3),
            "Mean_Abs_Change_cc": round(float(np.mean(np.abs(delta))), 3),
        })
    table2_cont = pd.DataFrame(continuity_rows)
    table2_cont.to_csv(os.path.join(OUT_DIR, "Table2_Longitudinal_Continuity.csv"), index=False)
    print("Table 2 (连续相关性):")
    print(table2_cont.to_string(index=False))

    # 状态转移矩阵 (0->0, 0->1, 1->0, 1->1)
    trans_rows = []
    for ep in ["V33", "V40", "Combined"]:
        col1 = f"CBCT1_{ep}_State" if ep != "Combined" else "CBCT1_Combined"
        col2 = f"CBCT2_{ep}_State" if ep != "Combined" else "CBCT2_Combined"
        tab = pd.crosstab(df_long[col1], df_long[col2])
        for s1 in [0, 1]:
            for s2 in [0, 1]:
                cnt = tab.loc[s1, s2] if (s1 in tab.index and s2 in tab.columns) else 0
                trans_rows.append({"Endpoint": ep, "Transition": f"{s1} -> {s2}", "Count": cnt, "Percent": round(cnt/len(df_long)*100, 1)})
    table2_trans = pd.DataFrame(trans_rows)
    table2_trans.to_csv(os.path.join(OUT_DIR, "Table2_State_Transitions.csv"), index=False)
    print("\nTable 2 (状态转移矩阵):")
    print(table2_trans.to_string(index=False))

    # 绘制 Figure 1: 纵向散点相关图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    for ax, k1, k2, name, t_val in [(ax1, "CBCT1_V33", "CBCT2_V33", "V33", t_v33), (ax2, "CBCT1_V40", "CBCT2_V40", "V40", t_v40)]:
        x, y = df_long[k1], df_long[k2]
        r, _ = stats.pearsonr(x, y)
        ax.scatter(x, y, color="#2980B9", alpha=0.8, edgecolors="k", s=55)
        lim = max(x.max(), y.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Identity Line (y=x)")
        ax.axvline(t_val, color="#E74C3C", linestyle=":", label=f"Threshold ({t_val:.1f}cc)")
        ax.axhline(t_val, color="#E74C3C", linestyle=":")
        ax.set_xlabel(f"CBCT 1 {name} (cc)", fontweight="bold", fontsize=11)
        ax.set_ylabel(f"CBCT 2 {name} (cc)", fontweight="bold", fontsize=11)
        ax.set_title(f"{name} Stability (r = {r:.3f})", fontweight="bold", fontsize=12)
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure1_Longitudinal_Scatter.png"))
    plt.close()

    # 绘制 Figure 2: 剂量差值分布直方图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    for ax, k1, k2, name in [(ax1, "CBCT1_V33", "CBCT2_V33", "V33"), (ax2, "CBCT1_V40", "CBCT2_V40", "V40")]:
        delta = df_long[k2] - df_long[k1]
        ax.hist(delta, bins=12, color="#34495E", edgecolor="k", alpha=0.75)
        ax.axvline(0, color="r", linestyle="--", linewidth=1.5)
        ax.set_xlabel(f"Δ{name} (CBCT2 - CBCT1, cc)", fontweight="bold", fontsize=11)
        ax.set_ylabel("Patient Count", fontweight="bold", fontsize=11)
        ax.set_title(f"Distribution of Δ{name} (Mean = {delta.mean():+.2f}cc)", fontweight="bold", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure2_Delta_Dose_Distributions.png"))
    plt.close()
    # 导出 Figure S1/S2 绘图数据
    df_long.to_csv(os.path.join(OUT_DIR, "Figure2_Longitudinal_Scatter_Data.csv"), index=False)
    print(f"  Figure2 散点数据已导出: Figure2_Longitudinal_Scatter_Data.csv")
    return df_long

# =============================================================================
# 模块 3: 严格患者级建模评估与奥卡姆剃刀验证 (TABLE 3, FIG 3 & 4)
# =============================================================================
def run_module_3(features_df, labels_df, df_long):
    print("\n" + "="*80)
    print(">>> [MODULE 3] 患者级预测模型评估 (奥卡姆剃刀法则验证: Table 3)")
    print("="*80)

    c1 = features_df[features_df["CBCT_ID"] == "CBCT_1"].set_index("Patient_ID")
    c2 = features_df[features_df["CBCT_ID"] == "CBCT_2"].set_index("Patient_ID")
    common_pids = sorted(list(set(c1.index) & set(c2.index)))
    
    # 构建患者级建模表
    target_series = c2.loc[common_pids, "Dosimetric_Risk"].astype(int)
    c1_patient = c1.loc[common_pids].copy()
    c1_patient["Target"] = target_series.values
    
    # 合并 Planning 物理特征
    plan_subset = labels_df.set_index("Patient_ID").loc[common_pids]
    c1_patient["Plan_V33"] = plan_subset["V33_Gastroduodenal_cc"].values
    c1_patient["Plan_V40"] = plan_subset["V40_Gastroduodenal_cc"].values
    c1_patient["Plan_Dmax"] = plan_subset["Max_Dose_Gy"].values
    c1_patient["Plan_Volume"] = plan_subset["Organ_Volume_cc"].values

    y = c1_patient["Target"].values

    # 定义 5 个核心模型对应的特征空间 (使用标准 L2 与 L1 兼顾，杜绝被强行杀成 0.5)
    models = {
        "M0: Planning Physical DVH": {
            "cols": ["Plan_V33", "Plan_V40", "Plan_Dmax", "Plan_Volume"],
            "penalty": "l2", "C": 1.0, "roi": "Planning Contours"
        },
        "M0b: CBCT1 Physical DVH": {
            "cols": ["V33_cc", "V40_cc"],
            "penalty": "l2", "C": 1.0, "roi": "CBCT1 OAR"
        },
        "M1: CBCT1 Whole-OAR Dosiomics": {
            "cols": [c for c in c1_patient.columns if c.startswith("WHOLE_")],
            "penalty": "l2", "C": 1.0, "roi": "Whole Irradiated Organ"
        },
        "M2: Dose-Gradient Dosiomics": {
            "cols": [c for c in c1_patient.columns if c.startswith("GRADIENT_") and "_U_" not in c],
            "penalty": "l2", "C": 1.0, "roi": "Penumbra (20%-80%)"
        },
        "M3: Gradient + Uncertainty Dosiomics": {
            "cols": [c for c in c1_patient.columns if c.startswith("GRADIENT_") or c.startswith("U_")],
            "penalty": "l2", "C": 1.0, "roi": "Penumbra + Uncertainty"
        }
    }

    results = {}
    table3_rows = []
    
    for m_id, m_cfg in models.items():
        sub_X = c1_patient[m_cfg["cols"]].copy()
        oof_probs = evaluate_model_pipeline(sub_X, y, penalty=m_cfg["penalty"], C=m_cfg["C"])
        auc_med, auc_ci, brier_med, boot_aucs = bootstrap_auc_ci(y, oof_probs)
        
        results[m_id] = {"oof": oof_probs, "boot_aucs": boot_aucs, "auc": auc_med}
        table3_rows.append({
            "Model_ID": m_id.split(":")[0],
            "Model_Description": m_id,
            "Input_Space": f"{len(m_cfg['cols'])} Features",
            "Target_ROI": m_cfg["roi"],
            "AUC_Median": round(auc_med, 4),
            "AUC_95CI": f"{auc_med:.3f} ({auc_ci[0]:.3f} - {auc_ci[1]:.3f})",
            "Brier_Score": round(brier_med, 4),
        })

    table3_df = pd.DataFrame(table3_rows)
    table3_df.to_csv(os.path.join(OUT_DIR, "Table3_Model_Comparison.csv"), index=False)
    print("Table 3 (模型对比):")
    print(table3_df.to_string(index=False))

    # 配对 ΔAUC 计算 (以极简物理基准 M0b 为基线)
    ref_boots = np.array(results["M0b: CBCT1 Physical DVH"]["boot_aucs"])
    paired_rows = []
    for m_id in models.keys():
        if m_id == "M0b: CBCT1 Physical DVH":
            continue
        cur_boots = np.array(results[m_id]["boot_aucs"])
        delta_boots = cur_boots - ref_boots
        p_val = float(min(1.0, 2.0 * min(np.mean(delta_boots >= 0), np.mean(delta_boots <= 0))))
        paired_rows.append({
            "Comparison": f"{m_id.split(':')[0]} vs M0b",
            "Delta_AUC_Median": round(float(np.median(delta_boots)), 4),
            "Delta_AUC_95CI": f"{np.median(delta_boots):.3f} ({np.percentile(delta_boots, 2.5):.3f} to {np.percentile(delta_boots, 97.5):.3f})",
            "P_Value": f"{p_val:.4f}"
        })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(OUT_DIR, "Table3_Paired_AUC_Delta.csv"), index=False)
    print("\nTable 3 (配对差异检验 vs M0b):")
    print(paired_df.to_string(index=False))
    # 导出 Figure S3 绘图数据
    roc_records = []
    for m_id, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["oof"])
        for f_val, t_val in zip(fpr, tpr):
            roc_records.append({"Model": m_id, "FPR": float(f_val), "TPR": float(t_val)})
    pd.DataFrame(roc_records).to_csv(
        os.path.join(OUT_DIR, "Figure3_ROC_Coordinates.csv"), index=False)
    print(f"  Figure3 ROC 数据已导出: Figure3_ROC_Coordinates.csv")
    # 绘制 Figure 3: ROC 曲线
    plt.figure(figsize=(7, 6), dpi=300)
    colors = ["#7F8C8D", "#E74C3C", "#2980B9", "#27AE60", "#8E44AD"]
    for (m_id, res), c in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y, res["oof"])
        plt.plot(fpr, tpr, color=c, lw=2.2, label=f"{m_id.split(':')[0]}: AUC={res['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    plt.xlabel("False Positive Rate", fontweight="bold", fontsize=11)
    plt.ylabel("True Positive Rate", fontweight="bold", fontsize=11)
    plt.title("Longitudinal Patient-Level ROC (CBCT1 -> CBCT2)", fontweight="bold", fontsize=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure3_Longitudinal_ROC.png"))
    plt.close()

    # 绘制 Figure 4: 模型 AUC 梯级柱状图
    plt.figure(figsize=(8.5, 5), dpi=300)
    bars = plt.bar(table3_df["Model_ID"], table3_df["AUC_Median"], color=colors, width=0.55, edgecolor="k")
    plt.axhline(0.5, color="k", linestyle="--", alpha=0.5)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Patient-Level AUC", fontweight="bold", fontsize=11)
    plt.title("Model Performance: Occam's Razor Demonstration", fontweight="bold", fontsize=12)
    for b, v in zip(bars, table3_df["AUC_Median"]):
        plt.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}", ha="center", fontweight="bold")
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "Figure4_Model_Comparison_Bars.png"))
    plt.close()

    return c1_patient

# =============================================================================
# 模块 4: 不确定性机制阴性审计 (TABLE 4)
# =============================================================================
def run_module_4(c1_patient, df_long):
    print("\n" + "="*80)
    print(">>> [MODULE 4] 机制阴性审计：不确定性与剂量变化关联分析 (Table 4)")
    print("="*80)

    u_cols = [c for c in ["U_Mean", "U_Median", "U_P75", "U_P90", "U_HighFraction"] if c in c1_patient.columns]
    merged = c1_patient[u_cols].merge(
        df_long.set_index("Patient_ID")[["CBCT1_V33", "CBCT2_V33", "CBCT1_V40", "CBCT2_V40"]],
        left_index=True, right_index=True
    )
    merged["Delta_V33"] = merged["CBCT2_V33"] - merged["CBCT1_V33"]
    merged["Delta_V40"] = merged["CBCT2_V40"] - merged["CBCT1_V40"]

    audit_rows = []
    for d_target in ["Delta_V33", "Delta_V40"]:
        y_val = merged[d_target].values
        for u in u_cols:
            x_val = merged[u].values
            r, p_pearson = stats.pearsonr(x_val, y_val)
            rho, p_spearman = stats.spearmanr(x_val, y_val)
            audit_rows.append({
                "Dosimetric_Drift": d_target,
                "Uncertainty_Metric": u,
                "Pearson_r": round(r, 4),
                "Pearson_p": round(p_pearson, 4),
                "Spearman_rho": round(rho, 4),
                "Spearman_p": round(p_spearman, 4),
            })

    table4 = pd.DataFrame(audit_rows)
    table4_path = os.path.join(OUT_DIR, "Table4_Uncertainty_vs_DeltaDose.csv")
    table4.to_csv(table4_path, index=False)
    print(f"Table 4 已保存: {table4_path}")
    print(table4.to_string(index=False))

# =============================================================================
# 主控程序入口
# =============================================================================
def main():
    print("="*90)
    print("MASTER EVALUATION & VALIDATION PIPELINE (ROUTE 1 DEFINITIVE)")
    print(f"Output Directory: {OUT_DIR}")
    print("="*90)

    assert os.path.exists(LABEL_FILE), f"Missing {LABEL_FILE}"
    assert os.path.exists(FEATURE_FILE), f"Missing {FEATURE_FILE}"

    labels_df = pd.read_csv(LABEL_FILE)
    features_df = pd.read_csv(FEATURE_FILE)

    t_v33, t_v40 = run_module_1(labels_df)
    df_long = run_module_2(features_df, t_v33, t_v40)
    c1_patient = run_module_3(features_df, labels_df, df_long)
    run_module_4(c1_patient, df_long)

    print("\n" + "="*90)
    print("ROUTE 1 MASTER EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"All Master Tables (Table 1~4) and Master Figures (Fig 1~4) are in: {OUT_DIR}")
    print("="*90)

if __name__ == "__main__":
    main()