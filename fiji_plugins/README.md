# Calcium Analysis Wizard v1-Manual - ImageJ/Fiji Plugin

A wizard-style ImageJ plugin for analyzing calcium transients in FGF2-G3 microscopy data to classify regenerating cells as cardiomyocytes or fibroblasts.

**Version**: v1-manual (manual ROI selection)

## Installation

### Prerequisites
- **Fiji** (ImageJ distribution) - Download from [fiji.sc](https://fiji.sc/)
- Bio-Formats plugin (included by default in Fiji)

### Install the Plugin

1. Copy the plugin file to the Fiji plugins folder:
   ```bash
   cp CalciumAnalysis_Wizard_v1_Manual.py /path/to/Fiji.app/plugins/
   ```

2. Restart Fiji

3. Access via: **Plugins > CalciumAnalysis Wizard v1 Manual**

## Usage

### Wizard Flow

```
File Selection → Preprocessing → Load & Merge (Diastole/Systole) → Adjust View → Select ROIs → Extract Traces → Export
```

### Key Feature: Diastole/Systole Visualization

The plugin automatically finds the **global minimum** (diastole) and **global maximum** (systole) calcium frames from the time-series. The merged image is a **4-channel x 2-timepoint hyperstack**:
- **T1**: Diastole (low calcium) - cells at rest
- **T2**: Systole (high calcium) - cells firing

Use the time slider to toggle between states and easily identify active cells!

### Channel Mapping (based on fluorophores)

| Channel | Fluorophore | LUT Color | Purpose |
|:--|:--|:--|:--|
| Ch1 | RFP | **Grays** | Actin (cell structure) |
| Ch2 | CFP | **Cyan** | G0/G1 nuclei (quiescent) |
| Ch3 | miRFP670 | **Magenta** | S/G2/M nuclei (regenerating) |
| Ch4 | Calcium indicator | **Green** | Calcium time-series |

### Step-by-Step

1. **File Selection**: Select static 3-channel image and calcium time-series
2. **Preprocessing** (NEW): Configure background subtraction and denoising
   - Background: Rolling Ball (default radius=50) or MOSAIC (if installed)
   - Denoising: Median filter (default radius=2) or Gaussian blur
   - Preview shows before/after comparison; adjust and re-preview as needed
3. **4-Channel Merge**: Plugin bins static image 4×, applies preprocessing, creates merged composite
4. **Adjust Visualization**: Use B&C and Channels tools to optimize view
5. **Select Magenta Cells**: Pick 5 S/G2/M cells (regenerating) with magenta nuclei
   - TIP: Turn off Cyan (Ch2) and Green (Ch4) to see magenta nuclei clearly
6. **Select Cyan Cells**: Pick 5 G0/G1 cells (quiescent) with cyan nuclei
   - TIP: Turn off Magenta (Ch3) and Green (Ch4) to see cyan nuclei clearly
7. **Select Background**: Pick 5 cell-free regions for background subtraction
8. **Trace Extraction**: Plugin measures intensity across all time-series frames
9. **Export**: Saves ROIs, traces, and classification summary

## Output Files

| File | Contents |
|:--|:--|
| `calcium_rois.zip` | ROI definitions (reloadable in ImageJ) |
| `calcium_traces_raw.csv` | Raw intensity values for all ROIs per frame |
| `calcium_traces_dff.csv` | F/F₀ normalized values + group statistics (mean, std, min, max) |
| `transient_summary.csv` | Cell classification results with group summary |

### Normalization

- **F₀ = min(trace)**: The minimum value of each trace is used as baseline
- **F/F₀**: Traces are normalized so baseline = 1.0, peaks > 1.0
- Best for spontaneously beating preparations where first frames may contain peaks

### Visualization

The plugin creates **3 plot windows**:
- **S/G2/M Regenerating**: Individual magenta cell traces + bold mean + min/max envelope
- **G0/G1 Quiescent**: Individual cyan cell traces + bold mean + min/max envelope
- **Background**: Background region traces + bold mean + min/max envelope

## Publication-Quality Figures (Python)

For publication-ready figures with shaded confidence intervals, use the included Python script:

```bash
# Install dependencies
pip install pandas numpy matplotlib

# Generate figure from exported CSV
python plot_traces.py --input /path/to/calcium_traces_dff.csv --output figure.png

# Options
python plot_traces.py -i traces.csv -o figure.pdf --ci sem   # SEM shading (default)
python plot_traces.py -i traces.csv -o figure.pdf --ci sd    # SD shading
python plot_traces.py -i traces.csv -o figure.png --dpi 600  # High resolution
```

## Classification Criteria

> ⚠️ **Work in Progress**: The peak detection algorithm is not yet optimized for spontaneously beating preparations. Classification thresholds below are preliminary and subject to refinement.

- **≥3 transients**: Likely Cardiomyocyte
- **1-2 transients**: Possible Cardiomyocyte
- **0 transients**: Likely Fibroblast

## Troubleshooting

- **Memory errors**: Time-series uses virtual stacks; increase Fiji memory if needed
- **Dimension mismatch**: Verify static image is 4× larger than time-series (2720 vs 680)
- **Channels not visible**: Use Channels Tool to toggle individual channels

## License

MIT License
