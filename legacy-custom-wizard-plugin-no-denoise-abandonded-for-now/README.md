# Calcium Analysis Regen

ImageJ/Fiji plugin for analyzing calcium transients in FGF2-G3 microscopy data to classify regenerating cells as cardiomyocytes or fibroblasts.

## Overview

This wizard-style plugin guides users through:
1. Loading static 3-channel images and calcium time-series
2. Creating a 4-channel merged composite for ROI selection
3. Selecting cells of interest (S/G2/M regenerating, G0/G1 quiescent)
4. Extracting calcium traces and detecting transients
5. Classifying cells based on calcium dynamics

## Installation

1. Download `fiji_plugins/CalciumAnalysis_Wizard.py`
2. Copy to your Fiji plugins folder: `Fiji.app/plugins/`
3. Restart Fiji
4. Run via **Plugins > CalciumAnalysis Wizard**

## Data Structure

Expected input:
- **Static 3-channel image** (2720×2720, .nd2)
  - Ch1: Actin (RFP)
  - Ch2: G0/G1 nuclei (CFP)
  - Ch3: S/G2/M nuclei (miRFP670)
- **Calcium time-series** (680×680, ~1000 frames, .nd2)

## Output

- `calcium_rois.zip` - ROI definitions
- `calcium_traces.csv` - ΔF/F₀ values
- `transient_summary.csv` - Cell classification

## License

MIT
