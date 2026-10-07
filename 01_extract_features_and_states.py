# -*- coding: utf-8 -*-
"""
================================================================================
01_EXTRACT_FEATURES_AND_STATES.PY (CONSOLIDATED STEP 1 & 2)
================================================================================
Core Functions:
1. Parse Planning CT + RTSTRUCT + RTDOSE -> Extract Stomach_duo baseline dose.
2. Establish fixed cohort median thresholds (V33 >= 65.807 cc, V40 >= 36.251 cc).
3. Extract longitudinal CBCT1/CBCT2 dose, OAR-specific gradient & uncertainty features.
4. Output standardized, clean CSVs for downstream evaluation.
================================================================================
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from radiomics import featureextractor
from skimage.draw import polygon
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("radiomics").setLevel(logging.ERROR)

# ----------------- 路径配置 -----------------
DATA_ROOT = r"C:\D盘数据转移\data剂量学衰退\manifest-1661266724052\Pancreatic-CT-CBCT-SEG"
DATA_DIR = r"C:\D盘数据转移\data胰腺癌毒性"
RESULT_DIR = r"D:\0临床科研\胰腺癌毒性\results\Evidential_Dosiomics"
os.makedirs(RESULT_DIR, exist_ok=True)

OUT_PLANNING_CSV = os.path.join(RESULT_DIR, "clinical_labels_detailed_v3.csv")
OUT_FEATURES_CSV = os.path.join(RESULT_DIR, "Longitudinal_DoseGradient_Features_v2.csv")

RADIOMICS_SETTINGS = {
    "binWidth": 5.0,
    "interpolator": "sitkBSpline",
    "resampledPixelSpacing": [2.0, 2.0, 2.0],
    "force2D": False,
    "normalize": False,
}

# ----------------- 基础工具函数 -----------------
def normalize_roi_name(name):
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "") if name else ""

def is_plan_roi(name):
    n = normalize_roi_name(name)
    return n in ["stomachduoplanct", "stomachduoplan", "stomachduo"] or ("stomachduo" in n and "cbct" not in n)

def safe_read_dicom(path):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None

def find_planning_ct_and_dose(patient_dir):
    ct_dirs, dose_files, struct_files = [], [], []
    for root, _, files in os.walk(patient_dir):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue
        ds = safe_read_dicom(os.path.join(root, dcm_files[0]))
        if ds is None:
            continue
        modality = str(getattr(ds, "Modality", "")).upper()
        if modality == "CT" and len(dcm_files) > 20:
            if "aligned" not in str(getattr(ds, "SeriesDescription", "")).lower():
                ct_dirs.append(root)
        elif modality == "RTDOSE":
            dose_files.extend([os.path.join(root, f) for f in dcm_files])
        elif modality == "RTSTRUCT":
            struct_files.extend([os.path.join(root, f) for f in dcm_files])

    if not ct_dirs:
        return None, None, struct_files

    # 排序锁定最早的 Planning CT
    candidates = []
    for d in ct_dirs:
        files = [f for f in os.listdir(d) if f.lower().endswith(".dcm")]
        ds = safe_read_dicom(os.path.join(d, files[0])) if files else None
        if ds:
            candidates.append((d, str(getattr(ds, "SeriesDate", "99999999")), str(getattr(ds, "SeriesTime", "000000"))))
    candidates.sort(key=lambda x: (x[1], x[2]))
    plan_ct_dir = candidates[0][0]

    # 锁定 RTDOSE
    dose_file = dose_files[0] if dose_files else None
    return plan_ct_dir, dose_file, struct_files

def find_plan_rtstruct(struct_files):
    for sf in struct_files:
        ds = safe_read_dicom(sf)
        if ds and str(getattr(ds, "Modality", "")).upper() == "RTSTRUCT":
            names = [str(getattr(roi, "ROIName", "")) for roi in getattr(ds, "StructureSetROISequence", [])]
            if any(is_plan_roi(n) for n in names):
                return sf
    return None

def rasterize_mask_native(rtstruct_path, ct_img):
    ds = pydicom.dcmread(rtstruct_path, force=True)
    W, H, D = ct_img.GetSize()
    mask = np.zeros((D, H, W), dtype=np.uint8)
    roi_numbers = set()
    for roi in getattr(ds, "StructureSetROISequence", []):
        if is_plan_roi(getattr(roi, "ROIName", "")):
            roi_numbers.add(int(roi.ROINumber))

    for c in getattr(ds, "ROIContourSequence", []):
        if getattr(c, "ReferencedROINumber", None) in roi_numbers:
            for s in getattr(c, "ContourSequence", []):
                pts = np.asarray(s.ContourData, dtype=np.float64).reshape(-1, 3)
                if len(pts) < 3:
                    continue
                idxs = [ct_img.TransformPhysicalPointToIndex(tuple(p)) for p in pts]
                z_vals = [idx[2] for idx in idxs]
                z_idx = int(round(float(np.median(z_vals))))
                if 0 <= z_idx < D:
                    rr, cc = polygon([np.clip(i[1], 0, H-1) for i in idxs], [np.clip(i[0], 0, W-1) for i in idxs], shape=(H, W))
                    mask[z_idx, rr, cc] = 1
    return mask

# ----------------- STEP 1: 规划期特征提取 -----------------
def extract_planning_cohort():
    print("\n>>> [STEP 1] 解析 Planning CT、Dose 与 Stomach_duo 轮廓...")
    patients = sorted([p for p in os.listdir(DATA_ROOT) if p.startswith("Pancreas") and os.path.isdir(os.path.join(DATA_ROOT, p))])
    records = []

    for pid in tqdm(patients, desc="Planning 解析"):
        p_dir = os.path.join(DATA_ROOT, pid)
        ct_dir, dose_file, struct_files = find_planning_ct_and_dose(p_dir)
        if not ct_dir or not dose_file:
            continue
        rtstruct = find_plan_rtstruct(struct_files)
        if not rtstruct:
            continue

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_dir))
        ct_img = reader.Execute()

        # 加载剂量并重采样至 CT
        ds_d = pydicom.dcmread(dose_file, stop_before_pixels=True, force=True)
        scale = float(getattr(ds_d, "DoseGridScaling", 1.0))
        dose_raw = sitk.ReadImage(dose_file)
        dose_arr = sitk.GetArrayFromImage(dose_raw).astype(np.float32) * scale
        dose_img = sitk.GetImageFromArray(dose_arr)
        dose_img.CopyInformation(dose_raw)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_img)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        dose_on_ct = sitk.GetArrayFromImage(resampler.Execute(dose_img))

        mask = rasterize_mask_native(rtstruct, ct_img)
        mask_vox = int(np.sum(mask > 0))
        if mask_vox < 32:
            continue

        vox_vol_cc = np.prod(ct_img.GetSpacing()) / 1000.0
        organ_vol = mask_vox * vox_vol_cc
        v33 = float(np.sum((dose_on_ct >= 33.0) & (mask > 0)) * vox_vol_cc)
        v40 = float(np.sum((dose_on_ct >= 40.0) & (mask > 0)) * vox_vol_cc)
        dmax = float(np.max(dose_on_ct[mask > 0]))

        records.append({
            "Patient_ID": pid,
            "Organ_Volume_cc": round(organ_vol, 3),
            "V33_Gastroduodenal_cc": round(v33, 3),
            "V40_Gastroduodenal_cc": round(v40, 3),
            "Max_Dose_Gy": round(dmax, 3),
        })

    df = pd.DataFrame(records)
    t_v33 = float(df["V33_Gastroduodenal_cc"].median())
    t_v40 = float(df["V40_Gastroduodenal_cc"].median())
    df["Label"] = ((df["V33_Gastroduodenal_cc"] >= t_v33) | (df["V40_Gastroduodenal_cc"] >= t_v40)).astype(int)
    df.to_csv(OUT_PLANNING_CSV, index=False, encoding="utf-8-sig")
    print(f"Planning 终点已锁定: V33 Threshold = {t_v33:.3f} cc, V40 Threshold = {t_v40:.3f} cc")
    return t_v33, t_v40

# ----------------- STEP 2: 纵向 CBCT 剂量组学提取 -----------------
def extract_longitudinal_features(t_v33, t_v40):
    print("\n>>> [STEP 2] 提取 CBCT1 & CBCT2 纵向梯度组学与不确定性特征...")
    patients = sorted([p for p in os.listdir(DATA_DIR) if p.startswith("Pancreas") and os.path.isdir(os.path.join(DATA_DIR, p))])
    all_features = []

    extractor = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_SETTINGS)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")

    for pid in tqdm(patients, desc="纵向特征提取"):
        p_dir = os.path.join(DATA_ROOT, pid)
        ct_dir, _, struct_files = find_planning_ct_and_dose(p_dir)
        rtstruct = find_plan_rtstruct(struct_files) if struct_files else None
        if not ct_dir or not rtstruct:
            continue

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_dir))
        native_ct = reader.Execute()

        # 生成 2mm 各向同性参考图
        target_sp = (2.0, 2.0, 2.0)
        orig_sp, orig_sz = native_ct.GetSpacing(), native_ct.GetSize()
        tgt_sz = [max(1, int(round(orig_sz[i] * orig_sp[i] / target_sp[i]))) for i in range(3)]
        res_ref = sitk.ResampleImageFilter()
        res_ref.SetSize(tgt_sz)
        res_ref.SetOutputSpacing(target_sp)
        res_ref.SetOutputOrigin(native_ct.GetOrigin())
        res_ref.SetOutputDirection(native_ct.GetDirection())
        ref_ct = res_ref.Execute(native_ct)

        native_mask = rasterize_mask_native(rtstruct, native_ct)
        m_img = sitk.GetImageFromArray(native_mask.astype(np.uint8))
        m_img.CopyInformation(native_ct)
        res_m = sitk.ResampleImageFilter()
        res_m.SetReferenceImage(ref_ct)
        res_m.SetInterpolator(sitk.sitkNearestNeighbor)
        oar_mask = (sitk.GetArrayFromImage(res_m.Execute(m_img)) > 0.5).astype(np.uint8)

        if oar_mask.sum() < 32:
            continue

        for cbct_id in ("CBCT_1", "CBCT_2"):
            npz_path = os.path.join(DATA_DIR, pid, cbct_id, "results.npz")
            if not os.path.exists(npz_path):
                continue
            data = np.load(npz_path, allow_pickle=True)
            dose = np.nan_to_num(data["warped_dose"].astype(np.float32))
            uncertainty = np.nan_to_num(data["uncertainty"].astype(np.float32))

            oar = oar_mask > 0
            dmax = float(np.max(dose[oar]))
            if dmax <= 1e-6:
                continue

            grad_mask = (oar & (dose >= 0.20 * dmax) & (dose <= 0.80 * dmax)).astype(np.uint8)
            vox_cc = 8.0 / 1000.0  # 2x2x2 mm^3 = 0.008 cc
            v33 = float(np.sum(dose[oar] >= 33.0) * vox_cc)
            v40 = float(np.sum(dose[oar] >= 40.0) * vox_cc)

            u_oar = uncertainty[oar]
            u_min, u_max = float(np.min(u_oar)), float(np.max(u_oar))
            u_norm = (uncertainty - u_min) / (u_max - u_min + 1e-8) if (u_max - u_min) >= 1e-8 else np.zeros_like(uncertainty)

            feat = {
                "Patient_ID": pid,
                "CBCT_ID": cbct_id,
                "V33_cc": v33,
                "V40_cc": v40,
                "OAR_Dmax_Gy": dmax,
                "OAR_Volume_cc": float(oar.sum() * vox_cc),
                "U_Mean": float(np.mean(u_norm[oar])),
                "U_Median": float(np.median(u_norm[oar])),
                "U_P75": float(np.percentile(u_norm[oar], 75)),
                "U_P90": float(np.percentile(u_norm[oar], 90)),
                "U_HighFraction": float(np.mean(u_norm[oar] >= 0.85)),
                "Dosimetric_Risk": int((v33 >= t_v33) or (v40 >= t_v40)),
            }

            # 提取全器官与梯度剂量组学
            sitk_dose = sitk.GetImageFromArray(dose)
            sitk_dose.SetSpacing(target_sp)
            for m_prefix, m_arr in [("WHOLE", oar_mask), ("GRADIENT", grad_mask)]:
                if m_arr.sum() >= 32:
                    sitk_m = sitk.GetImageFromArray(m_arr)
                    sitk_m.SetSpacing(target_sp)
                    res = extractor.execute(sitk_dose, sitk_m)
                    for k, v in res.items():
                        if not k.startswith("diagnostics"):
                            try:
                                feat[f"{m_prefix}_{k.replace('original_', '')}"] = float(v)
                            except Exception:
                                pass

            all_features.append(feat)

    df_feats = pd.DataFrame(all_features)
    df_feats.to_csv(OUT_FEATURES_CSV, index=False, encoding="utf-8-sig")
    print(f"纵向特征提取完成: 共保存 {len(df_feats)} 条记录 -> {OUT_FEATURES_CSV}")

if __name__ == "__main__":
    t_v33, t_v40 = extract_planning_cohort()
    extract_longitudinal_features(t_v33, t_v40)