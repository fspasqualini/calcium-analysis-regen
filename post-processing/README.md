# Post-processing

This folder contains scripts to generate montage videos and intensity/ROI plots
for the long-batch calcium imaging movies.

## Script: `make_montages.py`

Generates, for each raw movie in `training_data/long-batch/`:
- A 2x2 montage MP4 (raw + FAST + DeepCAD-RT + TeD)
- A global intensity (ΔF/F0) plot over time
- A ROI-averaged calcium transient plot (mean ± std across ROIs)
- ROI coordinates saved to JSON

### Default data locations
- Raw: `training_data/long-batch/`
- FAST: `results-from-FASTcode/phase1_default-*.zip`
- DeepCAD-RT: `results-from-DeepCAD/deepcad-rt-denoised/**/E_10_Iter_6350/`
- TeD: `results-from-TeD-cardioid-data/images/TeD_cardioid_long_batch/long-batch_epoch_99/`

### Example
```
python3 post-processing/make_montages.py \
  --raw-dir training_data/long-batch \
  --fast-zip-glob "results-from-FASTcode/phase1_default-*.zip" \
  --deepcad-dir "results-from-DeepCAD/deepcad-rt-denoised/*/E_10_Iter_6350" \
  --ted-dir "results-from-TeD-cardioid-data/images/TeD_cardioid_long_batch/long-batch_epoch_99" \
  --out-dir post-processing/output \
  --fps 10 \
  --roi-size 32 \
  --roi-count 5 \
  --seed 0

# Optional: run a single file
python3 post-processing/make_montages.py \
  --raw-dir training_data/long-batch \
  --fast-zip-glob "results-from-FASTcode/phase1_default-*.zip" \
  --deepcad-dir "results-from-DeepCAD/deepcad-rt-denoised/*/E_10_Iter_6350" \
  --ted-dir "results-from-TeD-cardioid-data/images/TeD_cardioid_long_batch/long-batch_epoch_99" \
  --out-dir post-processing/output \
  --only FGF2-G3_9.5110.tif
```

### Outputs (per movie)
- `montage.mp4`
- `global_dff.png`
- `roi_dff.png`
- `rois.json`
