# CalciumAnalysis_Utils.py
# Utility functions for the Calcium Analysis Wizard plugin
# Requires: Fiji with Bio-Formats

from ij import IJ, ImagePlus, CompositeImage
from ij.process import ImageProcessor, FloatProcessor
from ij.plugin.frame import RoiManager
from ij.measure import ResultsTable
from ij.gui import Plot, Roi
from java.awt import Color
import math

# -----------------------------------------------------------------------------
# Image Processing Utilities
# -----------------------------------------------------------------------------

def bin_image(imp, factor=4):
    """
    Bin an ImagePlus by the given factor using averaging.
    For a 3-channel composite, bins each channel separately.
    
    Args:
        imp: ImagePlus to bin
        factor: Binning factor (default 4)
    
    Returns:
        New ImagePlus with binned dimensions
    """
    original_width = imp.getWidth()
    original_height = imp.getHeight()
    new_width = original_width // factor
    new_height = original_height // factor
    n_channels = imp.getNChannels()
    n_slices = imp.getNSlices()
    n_frames = imp.getNFrames()
    
    # Create output stack
    from ij import ImageStack
    out_stack = ImageStack(new_width, new_height)
    
    stack = imp.getStack()
    for i in range(1, stack.getSize() + 1):
        ip = stack.getProcessor(i).convertToFloat()
        binned_ip = bin_processor(ip, factor)
        out_stack.addSlice(stack.getSliceLabel(i), binned_ip)
    
    result = ImagePlus("Binned_" + imp.getTitle(), out_stack)
    
    # Restore hyperstack dimensions if applicable
    if n_channels > 1 or n_slices > 1 or n_frames > 1:
        result.setDimensions(n_channels, n_slices, n_frames)
        if n_channels > 1:
            result = CompositeImage(result, CompositeImage.COMPOSITE)
    
    return result


def bin_processor(ip, factor):
    """
    Bin a single ImageProcessor by averaging pixels in factor x factor blocks.
    
    Args:
        ip: FloatProcessor to bin
        factor: Binning factor
    
    Returns:
        New FloatProcessor with binned dimensions
    """
    old_width = ip.getWidth()
    old_height = ip.getHeight()
    new_width = old_width // factor
    new_height = old_height // factor
    
    pixels = ip.getPixels()
    new_pixels = [0.0] * (new_width * new_height)
    
    for y in range(new_height):
        for x in range(new_width):
            total = 0.0
            for dy in range(factor):
                for dx in range(factor):
                    src_x = x * factor + dx
                    src_y = y * factor + dy
                    total += pixels[src_y * old_width + src_x]
            new_pixels[y * new_width + x] = total / (factor * factor)
    
    result = FloatProcessor(new_width, new_height, new_pixels)
    return result


def apply_composite_luts(imp, n_channels=4):
    """
    Apply standard LUTs to a composite image based on fluorophores:
    - Channel 1 (Actin): Grays (RFP)
    - Channel 2 (G0/G1 nuclei): Cyan (CFP)
    - Channel 3 (S/G2/M nuclei): Magenta (miRFP670)
    - Channel 4 (Calcium): Green (if present)
    
    Args:
        imp: CompositeImage
        n_channels: Number of channels (3 or 4)
    """
    if not isinstance(imp, CompositeImage):
        IJ.log("Warning: Image is not a CompositeImage, cannot apply LUTs")
        return
    
    imp.setDisplayMode(CompositeImage.COMPOSITE)
    
    # Channel 1: Grays (Actin - RFP)
    imp.setC(1)
    IJ.run(imp, "Grays", "")
    
    # Channel 2: Cyan (G0/G1 - CFP, quiescent)
    imp.setC(2)
    IJ.run(imp, "Cyan", "")
    
    # Channel 3: Magenta (S/G2/M - miRFP670, regenerating)
    imp.setC(3)
    IJ.run(imp, "Magenta", "")
    
    # Channel 4: Green (Calcium - if present)
    if n_channels >= 4:
        imp.setC(4)
        IJ.run(imp, "Green", "")
    
    # Reset to show all channels
    imp.setC(1)
    imp.updateAndDraw()


def auto_contrast(imp):
    """
    Apply auto-contrast to each channel of a composite image.
    Uses percentile-based stretching for better visualization of faint signals.
    
    Args:
        imp: CompositeImage to adjust
    """
    n_channels = imp.getNChannels()
    for c in range(1, n_channels + 1):
        imp.setC(c)
        IJ.run(imp, "Enhance Contrast", "saturated=0.35")
    imp.setC(1)
    imp.updateAndDraw()


# -----------------------------------------------------------------------------
# ROI Management Utilities
# -----------------------------------------------------------------------------

def get_roi_manager():
    """
    Get or create the ROI Manager instance.
    
    Returns:
        RoiManager instance
    """
    rm = RoiManager.getInstance()
    if rm is None:
        rm = RoiManager()
    return rm


def rename_rois(rm, prefix, start_index=0, count=5):
    """
    Rename the last 'count' ROIs in the ROI Manager with a prefix.
    
    Args:
        rm: RoiManager instance
        prefix: String prefix (e.g., "Cyan", "Magenta", "BG")
        start_index: Starting index for numbering
        count: Number of ROIs to rename
    """
    total_rois = rm.getCount()
    for i in range(count):
        roi_index = total_rois - count + i
        if roi_index >= 0:
            new_name = "%s_%d" % (prefix, i + 1)
            rm.rename(roi_index, new_name)


def validate_roi_count(rm, expected_count, roi_type):
    """
    Check if the ROI Manager has the expected number of new ROIs.
    
    Args:
        rm: RoiManager instance
        expected_count: Expected total count
        roi_type: String describing the ROI type for error messages
    
    Returns:
        True if count matches, False otherwise
    """
    actual_count = rm.getCount()
    if actual_count < expected_count:
        IJ.showMessage("ROI Count Error", 
            "Expected at least %d ROIs for %s, but only found %d.\n"
            "Please add more ROIs and try again." % (expected_count, roi_type, actual_count))
        return False
    return True


def save_rois(rm, filepath):
    """
    Save all ROIs in the ROI Manager to a ZIP file.
    
    Args:
        rm: RoiManager instance
        filepath: Path to save the ROI set
    """
    rm.runCommand("Deselect")
    rm.runCommand("Save", filepath)
    IJ.log("ROIs saved to: " + filepath)


# -----------------------------------------------------------------------------
# Trace Extraction and Analysis
# -----------------------------------------------------------------------------

def extract_traces(imp, rm):
    """
    Extract mean intensity traces from a time-series for all ROIs.
    
    Args:
        imp: ImagePlus time-series (single channel, multiple frames)
        rm: RoiManager with defined ROIs
    
    Returns:
        Dictionary with ROI names as keys and lists of intensities as values
    """
    n_frames = imp.getNFrames()
    if n_frames == 1:
        # Might be stored as slices
        n_frames = imp.getNSlices()
    
    n_rois = rm.getCount()
    traces = {}
    roi_names = []
    
    # Get ROI names
    for i in range(n_rois):
        name = rm.getName(i)
        roi_names.append(name)
        traces[name] = []
    
    IJ.log("Extracting traces for %d ROIs across %d frames..." % (n_rois, n_frames))
    
    # Extract intensities frame by frame
    for frame in range(1, n_frames + 1):
        if frame % 100 == 0:
            IJ.showProgress(frame, n_frames)
        
        imp.setSlice(frame)
        ip = imp.getProcessor()
        
        for i in range(n_rois):
            roi = rm.getRoi(i)
            imp.setRoi(roi)
            stats = ip.getStatistics()
            traces[roi_names[i]].append(stats.mean)
    
    IJ.showProgress(1.0)
    imp.killRoi()
    return traces


def calculate_dff(traces, baseline_frames=50, bg_rois=None):
    """
    Calculate Delta F / F0 for all traces.
    
    Args:
        traces: Dictionary of ROI name -> intensity list
        baseline_frames: Number of initial frames to use for F0 calculation
        bg_rois: List of background ROI names to average for background subtraction
    
    Returns:
        Dictionary of ROI name -> dF/F0 list
    """
    dff_traces = {}
    
    # Calculate background trace if specified
    bg_trace = None
    if bg_rois:
        bg_values = []
        for name in bg_rois:
            if name in traces:
                bg_values.append(traces[name])
        if bg_values:
            n_frames = len(bg_values[0])
            bg_trace = []
            for i in range(n_frames):
                avg = sum([bv[i] for bv in bg_values]) / len(bg_values)
                bg_trace.append(avg)
    
    for name, trace in traces.items():
        # Skip background ROIs in output
        if bg_rois and name in bg_rois:
            continue
        
        # Background subtraction
        if bg_trace:
            trace = [trace[i] - bg_trace[i] for i in range(len(trace))]
        
        # Calculate F0 (baseline)
        f0 = sum(trace[:baseline_frames]) / baseline_frames if baseline_frames > 0 else 1.0
        if f0 <= 0:
            f0 = 1.0  # Avoid division by zero
        
        # Calculate dF/F0
        dff = [(f - f0) / f0 for f in trace]
        dff_traces[name] = dff
    
    return dff_traces


def detect_transients(trace, min_prominence=0.1, min_height=0.05):
    """
    Simple peak detection for calcium transients.
    
    Args:
        trace: List of dF/F0 values
        min_prominence: Minimum prominence for peak detection
        min_height: Minimum height above baseline
    
    Returns:
        List of peak indices
    """
    peaks = []
    n = len(trace)
    
    for i in range(1, n - 1):
        # Simple local maximum detection
        if trace[i] > trace[i-1] and trace[i] > trace[i+1]:
            # Check height threshold
            if trace[i] > min_height:
                # Check prominence (difference from nearby minima)
                left_min = min(trace[max(0, i-20):i]) if i > 0 else trace[0]
                right_min = min(trace[i+1:min(n, i+21)]) if i < n-1 else trace[-1]
                prominence = trace[i] - max(left_min, right_min)
                
                if prominence > min_prominence:
                    peaks.append(i)
    
    return peaks


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_traces(dff_traces, title="Calcium Transients"):
    """
    Create a plot of all dF/F0 traces.
    
    Args:
        dff_traces: Dictionary of ROI name -> dF/F0 list
        title: Plot title
    
    Returns:
        Plot object
    """
    if not dff_traces:
        IJ.log("No traces to plot")
        return None
    
    # Get frame count from first trace
    first_trace = list(dff_traces.values())[0]
    n_frames = len(first_trace)
    x_values = list(range(n_frames))
    
    # Create plot
    plot = Plot(title, "Frame", "dF/F0")
    
    # Color coding by ROI type
    colors = {
        "Cyan": Color.CYAN,
        "Green": Color.GREEN,
        "Magenta": Color.MAGENTA,
        "BG": Color.GRAY
    }
    
    for name, trace in sorted(dff_traces.items()):
        # Determine color based on prefix
        color = Color.BLACK
        for prefix, c in colors.items():
            if name.startswith(prefix):
                color = c
                break
        
        plot.setColor(color)
        plot.addPoints(x_values, trace, Plot.LINE)
    
    # Add legend
    plot.setColor(Color.BLACK)
    plot.addLegend(" ".join(sorted(dff_traces.keys())))
    
    plot.show()
    return plot


def export_traces_csv(dff_traces, filepath):
    """
    Export traces to a CSV file.
    
    Args:
        dff_traces: Dictionary of ROI name -> dF/F0 list
        filepath: Output CSV path
    """
    if not dff_traces:
        IJ.log("No traces to export")
        return
    
    roi_names = sorted(dff_traces.keys())
    n_frames = len(list(dff_traces.values())[0])
    
    with open(filepath, 'w') as f:
        # Header
        f.write("Frame," + ",".join(roi_names) + "\n")
        
        # Data rows
        for i in range(n_frames):
            row = [str(i)]
            for name in roi_names:
                row.append("%.6f" % dff_traces[name][i])
            f.write(",".join(row) + "\n")
    
    IJ.log("Traces exported to: " + filepath)


def generate_summary(dff_traces, output_path):
    """
    Generate a summary of transient detection for each ROI.
    
    Args:
        dff_traces: Dictionary of ROI name -> dF/F0 list
        output_path: Path for summary CSV
    """
    with open(output_path, 'w') as f:
        f.write("ROI,Type,NumPeaks,Classification\n")
        
        for name, trace in sorted(dff_traces.items()):
            # Determine cell type from name (based on ROI prefix)
            if name.startswith("Magenta"):
                cell_type = "S/G2/M (Regenerating)"
            elif name.startswith("Cyan"):
                cell_type = "G0/G1 (Quiescent)"
            else:
                cell_type = "Unknown"
            
            # Detect transients
            peaks = detect_transients(trace)
            n_peaks = len(peaks)
            
            # Classify based on peak count
            if n_peaks >= 3:
                classification = "Likely Cardiomyocyte"
            elif n_peaks >= 1:
                classification = "Possible Cardiomyocyte"
            else:
                classification = "Likely Fibroblast"
            
            f.write("%s,%s,%d,%s\n" % (name, cell_type, n_peaks, classification))
    
    IJ.log("Summary exported to: " + output_path)
