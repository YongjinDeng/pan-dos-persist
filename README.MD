# Longitudinal Dosimetric Persistence in Pancreatic Adaptive Radiotherapy

Analysis code for the manuscript:

> Longitudinal Dosimetric Persistence in Pancreatic Adaptive Radiotherapy: Simple Physical Dose-Volume Metrics Outperform High-Dimensional Dosiomics and Registration Uncertainty

## Data

Imaging data are publicly available from The Cancer Imaging Archive (TCIA):
Pancreatic-CT-CBCT-SEG (DOI: 10.7937/TCIA.ESHQ-4D90)

## Code

| File | Description |
|------|-------------|
| `00_PI_INR_Retrain.py` | PI-INR deformable image registration and dose pullback |
| `01_extract_features_and_states.py` | Planning CT parsing and dosiomic feature extraction |
| `02_run_master_evaluation.py` | Longitudinal continuity analysis and model benchmarking |
| `03_generate_supp_figures.py` | Supplementary figure generation |

## Requirements

```
pip install torch SimpleITK pydicom pyradiomics scikit-learn pandas numpy matplotlib scipy tqdm
```

## Usage

```bash
python 00_PI_INR_Retrain.py
python 01_extract_features_and_states.py
python 02_run_master_evaluation.py
python 03_generate_supp_figures.py
```

## Output

All results are saved to `Route1_Master_Results/`:
- `Table1_Cohort_Characteristics.csv`
- `Table2_Longitudinal_Continuity.csv`
- `Table2_State_Transitions.csv`
- `Table3_Model_Comparison.csv`
- `Table3_Paired_AUC_Delta.csv`
- `Table4_Uncertainty_vs_DeltaDose.csv`
- `Figure1_Longitudinal_Scatter.png`
- `Figure2_Delta_Dose_Distributions.png`
- `Figure3_Longitudinal_ROC.png`
- `Figure4_Model_Comparison_Bars.png`

## License

MIT License
