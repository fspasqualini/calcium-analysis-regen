# Calcium Analysis Wizard - ImageJ/Fiji Plugin

A wizard-style ImageJ plugin for analyzing calcium transients in FGF2-G3 microscopy data to classify regenerating cells as cardiomyocytes or fibroblasts.

## Installation

### Prerequisites
- **Fiji** (ImageJ distribution) - Download from [fiji.sc](https://fiji.sc/)
- Bio-Formats plugin (included by default in Fiji)

### Install the Plugin

1. Copy both plugin files to the Fiji plugins folder:
   ```bash
   cp CalciumAnalysis_Wizard.py /path/to/Fiji.app/plugins/
   cp CalciumAnalysis_Utils.py /path/to/Fiji.app/plugins/
   ```

2. Restart Fiji

3. Access via: **Plugins > CalciumAnalysis Wizard**

## Usage

### Wizard Flow

```
File Selection → Load & Merge → Adjust View → Select ROIs → Extract Traces → Export
```

### Channel Mapping (based on fluorophores)

| Channel | Fluorophore | LUT Color | Purpose |
|:--|:--|:--|:--|
| Ch1 | RFP | **Grays** | Actin (cell structure) |
| Ch2 | CFP | **Cyan** | G0/G1 nuclei (quiescent) |
| Ch3 | miRFP670 | **Magenta** | S/G2/M nuclei (regenerating) |
| Ch4 | Calcium indicator | **Green** | Calcium time-series |

### Step-by-Step

1. **File Selection**: Select static 3-channel image and calcium time-series
2. **4-Channel Merge**: Plugin bins static image 4×, loads time-series, creates merged composite
3. **Adjust Visualization**: Use B&C and Channels tools to optimize view
4. **Select Magenta Cells**: Pick 5 S/G2/M cells (regenerating) with magenta nuclei
   - TIP: Turn off Cyan (Ch2) and Green (Ch4) to see magenta nuclei clearly
5. **Select Cyan Cells**: Pick 5 G0/G1 cells (quiescent) with cyan nuclei
   - TIP: Turn off Magenta (Ch3) and Green (Ch4) to see cyan nuclei clearly
6. **Select Background**: Pick 5 cell-free regions for background subtraction
7. **Trace Extraction**: Plugin measures intensity across all time-series frames
8. **Export**: Saves ROIs, traces, and classification summary

## Output Files

| File | Contents |
|:--|:--|
| `calcium_rois.zip` | ROI definitions (reloadable in ImageJ) |
| `calcium_traces.csv` | ΔF/F₀ values for all ROIs per frame |
| `transient_summary.csv` | Cell classification results |

## Classification Criteria

- **≥3 transients**: Likely Cardiomyocyte
- **1-2 transients**: Possible Cardiomyocyte
- **0 transients**: Likely Fibroblast

## Troubleshooting

- **Memory errors**: Time-series uses virtual stacks; increase Fiji memory if needed
- **Dimension mismatch**: Verify static image is 4× larger than time-series (2720 vs 680)
- **Channels not visible**: Use Channels Tool to toggle individual channels

## License

MIT License
