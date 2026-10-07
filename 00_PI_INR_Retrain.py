# 01_PI_INR_Retrain.py
# -*- coding: utf-8 -*-
r"""
PI-INR 重训版 v2：
- 每个 CBCT 独立从零训练（无热启动）
- 只遍历 Aligned CBCT
- 规划 CT 识别：任何非 Aligned 的 CT，按时间排序取最早
- CBCT 命名统一：按 SeriesDate 排序，CBCT_1 (早) / CBCT_2 (晚)
- 输出：C:\D盘数据转移\data胰腺癌毒性\<患者ID>\CBCT_1|CBCT_2\results.npz
- 支持断点续跑
"""

import os
import gc
import time
import torch
import pydicom
import numpy as np
import SimpleITK as sitk
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ================== 1. 配置 ==================
CONFIG = {
    "DATA_ROOT": r"C:\D盘数据转移\data剂量学衰退\manifest-1661266724052\Pancreatic-CT-CBCT-SEG",
    "RESULT_DIR": r"C:\D盘数据转移\data胰腺癌毒性",

    "NUM_THREADS": max(1, (os.cpu_count() or 4) - 1),
    "EPOCHS": 800,
    "BATCH_POINTS": 15000,
    "LR": 5e-4,
    "OMEGA_0": 20.0,

    "LAMBDA_EDL": 1.0,
    "LAMBDA_EDGE": 5.0,
    "LAMBDA_SMOOTH": 0.1,
    "LAMBDA_PHYS": 0.2,
    "LAMBDA_FOLD_INIT": 0.1,
    "LAMBDA_FOLD_MAX": 5.0,
    "GAMMA_METRIC": 10.0,

    "TARGET_SPACING": (2.0, 2.0, 2.0),
    "RANDOM_SEED": 42,

    # 只跑前 N 个患者（调试用，None=全部）
    "DEBUG_PATIENT_LIMIT": None,
}

torch.set_num_threads(CONFIG["NUM_THREADS"])
torch.set_flush_denormal(True)
os.environ["OMP_NUM_THREADS"] = str(CONFIG["NUM_THREADS"])
os.environ["MKL_NUM_THREADS"] = str(CONFIG["NUM_THREADS"])

CONFIG["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(CONFIG["RESULT_DIR"], exist_ok=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["RANDOM_SEED"])


# ================== 2. 网络 ==================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=20.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / in_features) / self.omega_0,
                    np.sqrt(6 / in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class RiemannianSirenNet(nn.Module):
    def __init__(self, omega_0=20.0):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(3, 128, is_first=True, omega_0=omega_0),
            SineLayer(128, 128, omega_0=omega_0),
            SineLayer(128, 128, omega_0=omega_0),
            nn.Linear(128, 6)
        )
        with torch.no_grad():
            self.net[-1].weight.fill_(0)
            self.net[-1].bias.fill_(0)

    def forward(self, x):
        out = self.net(x)
        disp = torch.tanh(out[..., :3]) * 0.05
        v = F.softplus(out[..., 3:4]) + 1e-6
        alpha = F.softplus(out[..., 4:5]) + 1.1
        beta = F.softplus(out[..., 5:6]) + 1e-6
        return disp, v.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)


# ================== 3. 工具 ==================
def compute_image_gradients(img_tensor):
    dz = img_tensor[..., 2:, 1:-1, 1:-1] - img_tensor[..., :-2, 1:-1, 1:-1]
    dy = img_tensor[..., 1:-1, 2:, 1:-1] - img_tensor[..., 1:-1, :-2, 1:-1]
    dx = img_tensor[..., 1:-1, 1:-1, 2:] - img_tensor[..., 1:-1, 1:-1, :-2]
    mag = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + 1e-6)
    return F.pad(mag, (1, 1, 1, 1, 1, 1), mode='replicate')


def fast_stratified_sample(batch_size, edge_idx, bg_idx, fixed_flat, fixed_edges_flat, shape, device):
    D, H, W = shape
    e_batch = int(batch_size * 0.7)
    b_batch = batch_size - e_batch

    if len(edge_idx) > 0:
        s_edge = edge_idx[torch.randint(0, len(edge_idx), (e_batch,), device=device)]
    else:
        s_edge = bg_idx[torch.randint(0, len(bg_idx), (e_batch,), device=device)]

    if len(bg_idx) > 0:
        s_bg = bg_idx[torch.randint(0, len(bg_idx), (b_batch,), device=device)]
    else:
        s_bg = edge_idx[torch.randint(0, len(edge_idx), (b_batch,), device=device)]

    idx = torch.cat([s_edge, s_bg])
    f_s = fixed_flat[idx]
    fe_s = fixed_edges_flat[idx]

    z = idx // (H * W)
    y = (idx % (H * W)) // W
    x = idx % W

    coords = torch.stack([x, y, z], dim=1).float()
    sizes = torch.tensor([W - 1, H - 1, D - 1], device=device).float()
    coords = (coords / sizes) * 2.0 - 1.0
    return coords.requires_grad_(True), f_s, fe_s


def compute_jacobian_fast(y, x, grad_outputs_list):
    jac = []
    for i in range(3):
        grad = torch.autograd.grad(y[:, i], x, grad_outputs=grad_outputs_list[i],
                                   create_graph=True, retain_graph=True)[0]
        jac.append(grad)
    return torch.stack(jac, dim=1)


def compute_fold_loss(J, I_mat, device):
    det = torch.det(I_mat + J)
    det_clamped = torch.clamp(det, min=-10.0, max=10.0)
    fold_loss = F.relu(-det_clamped + 1e-5).mean()
    expansion_penalty = torch.mean((det_clamped - 1.0) ** 2) * 0.001
    return fold_loss + expansion_penalty


# ================== 4. 数据加载 ==================
def resample_to_spacing(image, target_spacing, default_value=0):
    orig_spacing, orig_size = image.GetSpacing(), image.GetSize()
    target_size = [max(1, int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(image)


def scan_ct_series(patient_path):
    """扫描所有 CT series，返回 [(目录, SeriesDate, SeriesTime, SeriesDescription), ...]"""
    results = []
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_files or len(dcm_files) < 20:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(root, dcm_files[0]), stop_before_pixels=True)
            if getattr(ds, 'Modality', '').upper() == 'CT':
                results.append((
                    root,
                    getattr(ds, 'SeriesDate', '99999999'),
                    getattr(ds, 'SeriesTime', '000000'),
                    getattr(ds, 'SeriesDescription', '')
                ))
        except:
            continue
    return results


def find_planning_ct(patient_path):
    """
    找规划 CT：任何非 Aligned 的 CT，按 SeriesDate 排序取最早的。
    修订版：不再要求 iDose/PANCREAS，避免漏掉描述不同的规划 CT。
    """
    all_series = scan_ct_series(patient_path)
    candidates = [s for s in all_series if 'Aligned' not in s[3]]
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], x[2]))
    return candidates[0][0]


def find_cbct_series(patient_path):
    """找所有 Aligned CBCT series，按时间排序"""
    all_series = scan_ct_series(patient_path)
    cbcts = [s for s in all_series if 'Aligned' in s[3]]
    cbcts.sort(key=lambda x: (x[1], x[2]))
    return cbcts


def load_planning_ct(patient_path, target_spacing):
    fixed_dir = find_planning_ct(patient_path)
    if fixed_dir is None:
        return None

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(fixed_dir))
    fixed = sitk.Cast(reader.Execute(), sitk.sitkFloat32)

    arr = np.nan_to_num(sitk.GetArrayFromImage(fixed), nan=-1000)
    fixed_s = sitk.GetImageFromArray(arr)
    fixed_s.CopyInformation(fixed)

    fixed_rs = resample_to_spacing(fixed_s, target_spacing, -1000)

    arr_norm = np.clip((sitk.GetArrayFromImage(fixed_rs).astype(np.float32) + 1000) / 2000.0, 0, 1)
    fixed_t = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).to(CONFIG["DEVICE"])

    return fixed_t, fixed_rs


def load_dose(patient_path, fixed_sitk):
    dose_path = None
    for root, dirs, files in os.walk(patient_path):
        for f in files:
            if f.lower().endswith('.dcm'):
                try:
                    ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                    if getattr(ds, 'Modality', '').upper() == 'RTDOSE':
                        dose_path = os.path.join(root, f)
                        break
                except:
                    continue
        if dose_path:
            break
    if dose_path is None:
        return None

    ds_d = pydicom.dcmread(dose_path, stop_before_pixels=True)
    scale = float(getattr(ds_d, 'DoseGridScaling', 1.0))

    dose_raw = sitk.ReadImage(dose_path)
    dose_arr = sitk.GetArrayFromImage(dose_raw).astype(np.float32) * scale
    dose_gy = sitk.GetImageFromArray(dose_arr)
    dose_gy.CopyInformation(dose_raw)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(sitk.Transform())
    dose_on_fixed = resampler.Execute(dose_gy)

    dose_arr = np.clip(sitk.GetArrayFromImage(dose_on_fixed), 0, 100)
    dose_t = torch.from_numpy(dose_arr).unsqueeze(0).unsqueeze(0).to(CONFIG["DEVICE"])
    return dose_t


def load_moving_cbct(cbct_dir, fixed_sitk, target_spacing):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(cbct_dir))
    moving_raw = sitk.Cast(reader.Execute(), sitk.sitkFloat32)

    arr = np.nan_to_num(sitk.GetArrayFromImage(moving_raw), nan=-1000)
    moving_s = sitk.GetImageFromArray(arr)
    moving_s.CopyInformation(moving_raw)

    fixed_center = np.array(fixed_sitk.TransformContinuousIndexToPhysicalPoint(
        [s / 2.0 for s in fixed_sitk.GetSize()]))
    moving_center = np.array(moving_s.TransformContinuousIndexToPhysicalPoint(
        [s / 2.0 for s in moving_s.GetSize()]))
    translation = (moving_center - fixed_center).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetTransform(sitk.TranslationTransform(3, translation))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)
    moving_aligned = resampler.Execute(moving_s)
    moving_rs = resample_to_spacing(moving_aligned, target_spacing, -1000)

    arr_norm = np.clip((sitk.GetArrayFromImage(moving_rs).astype(np.float32) + 1000) / 2000.0, 0, 1)
    moving_t = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).to(CONFIG["DEVICE"])
    return moving_t


# ================== 5. 训练 ==================
def register_cbct(model, fixed_t, moving_t, dose_t, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-6)

    D, H, W = fixed_t.shape[2:]

    try:
        spacing = [CONFIG["TARGET_SPACING"][2], CONFIG["TARGET_SPACING"][1], CONFIG["TARGET_SPACING"][0]]
        dz, dy, dx = torch.gradient(dose_t.squeeze(), spacing=spacing)
        grad_dose = torch.stack([dx, dy, dz], dim=0).unsqueeze(0).to(CONFIG["DEVICE"])
    except:
        grad_dose = torch.zeros(1, 3, D, H, W, device=CONFIG["DEVICE"])

    fixed_edges = compute_image_gradients(fixed_t)
    fixed_flat = fixed_t.view(-1)
    fixed_edges_flat = fixed_edges.view(-1)
    edge_threshold = fixed_edges_flat.mean()

    edge_idx = torch.nonzero(fixed_edges_flat > edge_threshold).squeeze()
    bg_idx = torch.nonzero(fixed_edges_flat <= edge_threshold).squeeze()
    if edge_idx.dim() == 0:
        edge_idx = edge_idx.unsqueeze(0)
    if bg_idx.dim() == 0:
        bg_idx = bg_idx.unsqueeze(0)

    I_mat = torch.eye(3, device=CONFIG["DEVICE"]).unsqueeze(0)
    grad_outputs_list = [torch.ones(CONFIG["BATCH_POINTS"], device=CONFIG["DEVICE"]) for _ in range(3)]

    pbar = tqdm(range(epochs), desc="  Training", leave=False)
    for epoch in pbar:
        opt.zero_grad()

        coords, f_s, fe_s = fast_stratified_sample(
            CONFIG["BATCH_POINTS"], edge_idx, bg_idx,
            fixed_flat, fixed_edges_flat, (D, H, W), CONFIG["DEVICE"])

        disp, v, alpha, beta = model(coords)

        m_s = F.grid_sample(moving_t, (coords + disp).view(1, 1, 1, -1, 3),
                            align_corners=True, mode='bilinear').view(-1)
        me_s = F.grid_sample(compute_image_gradients(moving_t), (coords + disp).view(1, 1, 1, -1, 3),
                             align_corners=True, mode='bilinear').view(-1)

        mask = (f_s > 0.05).float()
        edl_loss = torch.mean(mask * ((f_s - m_s) ** 2 * v + (2 * alpha + v) / (2 * alpha * v)))
        edge_loss = torch.mean(mask * (fe_s - me_s) ** 2)

        try:
            J = compute_jacobian_fast(disp, coords, grad_outputs_list)
            s_grad = F.grid_sample(grad_dose, coords.view(1, 1, 1, -1, 3),
                                   align_corners=True).view(3, -1).T
            phys_loss = torch.mean(
                torch.clamp(1.0 + CONFIG["GAMMA_METRIC"] * torch.sum(s_grad ** 2, dim=-1), 1.0, 50.0)
                * torch.sum(J ** 2, dim=(1, 2)))
            fold_loss = compute_fold_loss(J, I_mat, CONFIG["DEVICE"])
        except:
            phys_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])
            fold_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])

        try:
            grad_disp = torch.autograd.grad(disp.sum(), coords, create_graph=True)[0]
            smooth_loss = torch.mean(grad_disp ** 2)
        except:
            smooth_loss = torch.tensor(0.0, device=CONFIG["DEVICE"])

        progress = epoch / epochs
        w_fold = CONFIG["LAMBDA_FOLD_INIT"] + \
                 (CONFIG["LAMBDA_FOLD_MAX"] - CONFIG["LAMBDA_FOLD_INIT"]) * (progress ** 0.5)

        total_loss = (edl_loss
                      + CONFIG["LAMBDA_EDGE"] * edge_loss
                      + CONFIG["LAMBDA_SMOOTH"] * smooth_loss
                      + CONFIG["LAMBDA_PHYS"] * phys_loss
                      + w_fold * fold_loss)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if epoch % 100 == 0:
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

    return model


# ================== 6. 推理 ==================
def inference_full(model, fixed_t, moving_t, dose_t, D, H, W, spacing):
    model.eval()
    displacement = np.zeros((D, H, W, 3), dtype=np.float32)
    uncertainty = np.zeros((D, H, W), dtype=np.float32)

    with torch.no_grad():
        for z in tqdm(range(D), desc="  Inference", leave=False):
            z_n = (z / (D - 1)) * 2 - 1 if D > 1 else 0
            gy, gx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=CONFIG["DEVICE"]),
                torch.linspace(-1, 1, W, device=CONFIG["DEVICE"]),
                indexing='ij')
            grid_slice = torch.stack([gx, gy, torch.full_like(gx, z_n)], dim=-1).view(-1, 3)
            disp_s, v_s, alpha_s, beta_s = model(grid_slice)
            displacement[z] = disp_s.view(H, W, 3).cpu().numpy()
            uncertainty[z] = (beta_s / (v_s * (alpha_s - 1))).view(H, W).cpu().numpy()

        z_coords = torch.linspace(-1, 1, D, device=CONFIG["DEVICE"])
        y_coords = torch.linspace(-1, 1, H, device=CONFIG["DEVICE"])
        x_coords = torch.linspace(-1, 1, W, device=CONFIG["DEVICE"])
        gz, gy, gx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        base_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0)

        disp_tensor = torch.from_numpy(displacement).permute(3, 0, 1, 2).unsqueeze(0).to(CONFIG["DEVICE"])
        warped_ct = F.grid_sample(moving_t, base_grid + disp_tensor.permute(0, 2, 3, 4, 1),
                                  align_corners=True, mode='bilinear').squeeze().cpu().numpy()
        warped_dose = F.grid_sample(dose_t, base_grid + disp_tensor.permute(0, 2, 3, 4, 1),
                                    align_corners=True, mode='bilinear').squeeze().cpu().numpy()

    sizes = np.array([W - 1, H - 1, D - 1], dtype=np.float32)
    sp = np.array([spacing[0], spacing[1], spacing[2]], dtype=np.float32)
    disp_mm = displacement * sizes / 2.0 * sp

    return displacement, uncertainty, warped_ct, warped_dose, disp_mm


# ================== 7. 主流程 ==================
def main():
    print("=" * 72)
    print("PI-INR 重训版 v2 (每个 CBCT 独立训练，命名统一)")
    print(f"  Device: {CONFIG['DEVICE']}")
    print(f"  Threads: {CONFIG['NUM_THREADS']}")
    print(f"  Epochs: {CONFIG['EPOCHS']}")
    print(f"  Batch: {CONFIG['BATCH_POINTS']}")
    print(f"  Data: {CONFIG['DATA_ROOT']}")
    print(f"  Output: {CONFIG['RESULT_DIR']}")
    print("=" * 72)

    if not os.path.exists(CONFIG["DATA_ROOT"]):
        print(f"❌ 数据路径不存在: {CONFIG['DATA_ROOT']}")
        return

    all_patients = sorted([p for p in os.listdir(CONFIG["DATA_ROOT"])
                           if p.startswith("Pancreas") and
                           os.path.isdir(os.path.join(CONFIG["DATA_ROOT"], p))])
    if CONFIG["DEBUG_PATIENT_LIMIT"]:
        all_patients = all_patients[:CONFIG["DEBUG_PATIENT_LIMIT"]]

    print(f"发现 {len(all_patients)} 个患者\n")

    total_start = time.time()
    done_count = 0
    skip_count = 0
    fail_count = 0

    for pid_idx, pid in enumerate(all_patients):
        patient_path = os.path.join(CONFIG["DATA_ROOT"], pid)
        save_root = os.path.join(CONFIG["RESULT_DIR"], pid)
        os.makedirs(save_root, exist_ok=True)

        print(f"\n{'=' * 72}")
        print(f"📂 [{pid_idx+1}/{len(all_patients)}] {pid}")
        print(f"{'=' * 72}")

        fixed_data = load_planning_ct(patient_path, CONFIG["TARGET_SPACING"])
        if fixed_data is None:
            print(f"⚠️ 规划 CT 加载失败，跳过")
            fail_count += 1
            continue
        fixed_t, fixed_sitk = fixed_data
        D, H, W = fixed_t.shape[2:]
        print(f"  图像尺寸: D={D}, H={H}, W={W}")

        dose_t = load_dose(patient_path, fixed_sitk)
        if dose_t is None:
            print(f"⚠️ RTDOSE 加载失败，跳过")
            fail_count += 1
            continue

        cbcts = find_cbct_series(patient_path)
        print(f"  找到 {len(cbcts)} 个 Aligned CBCT")
        for cbct_dir, date, time_, desc in cbcts:
            print(f"     - {desc} | {date} {time_}")

        # 统一命名：按 SeriesDate 排序后 CBCT_1 (早) / CBCT_2 (晚)
        for cbct_idx, (cbct_dir, cbct_date, cbct_time, cbct_desc) in enumerate(cbcts):
            fr_id = f"CBCT_{cbct_idx+1}"   # 统一命名

            fr_dir = os.path.join(save_root, fr_id)
            os.makedirs(fr_dir, exist_ok=True)

            result_path = os.path.join(fr_dir, "results.npz")
            if os.path.exists(result_path) and os.path.getsize(result_path) > 50 * 1024 * 1024:
                print(f"  ⏭️ {fr_id} 已存在 ({os.path.getsize(result_path)/1024/1024:.1f} MB)，跳过")
                skip_count += 1
                continue

            print(f"\n  🔄 {fr_id} ({cbct_desc}) ...")
            t0 = time.time()
            try:
                moving_t = load_moving_cbct(cbct_dir, fixed_sitk, CONFIG["TARGET_SPACING"])
                model = RiemannianSirenNet(omega_0=CONFIG["OMEGA_0"]).to(CONFIG["DEVICE"])
                model = register_cbct(model, fixed_t, moving_t, dose_t, CONFIG["EPOCHS"])

                displacement, uncertainty, warped_ct, warped_dose, disp_mm = inference_full(
                    model, fixed_t, moving_t, dose_t, D, H, W, CONFIG["TARGET_SPACING"])

                np.savez_compressed(
                    result_path,
                    displacement=displacement.astype(np.float16),
                    displacement_physical_mm=disp_mm.astype(np.float16),
                    uncertainty=uncertainty.astype(np.float16),
                    warped_ct=warped_ct.astype(np.float16),
                    warped_dose=warped_dose.astype(np.float16),
                    shape=(D, H, W),
                    spacing=CONFIG["TARGET_SPACING"],
                    source_cbct=cbct_dir,
                    source_date=cbct_date,
                    source_time=cbct_time)

                elapsed = time.time() - t0
                print(f"     ✅ 保存: {result_path}  (耗时 {elapsed/60:.1f} 分钟)")
                done_count += 1

                del model, moving_t
                gc.collect()
            except Exception as e:
                print(f"     ❌ 失败: {type(e).__name__}: {e}")
                fail_count += 1
                continue

        del fixed_t, fixed_sitk, dose_t
        gc.collect()

        elapsed_total = time.time() - total_start
        avg = elapsed_total / (pid_idx + 1)
        remaining = avg * (len(all_patients) - pid_idx - 1)
        print(f"\n  ⏱️ 累计 {elapsed_total/3600:.2f} h，预计剩余 {remaining/3600:.2f} h")
        print(f"     完成 {done_count} / 跳过 {skip_count} / 失败 {fail_count}")

    print("\n" + "=" * 72)
    print(f"✅ 全部完成！总耗时 {(time.time()-total_start)/3600:.2f} 小时")
    print(f"   成功 {done_count} / 跳过 {skip_count} / 失败 {fail_count}")
    print(f"   输出目录: {CONFIG['RESULT_DIR']}")
    print("=" * 72)


if __name__ == "__main__":
    main()