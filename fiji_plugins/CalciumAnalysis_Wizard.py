# CalciumAnalysis_Wizard.py
# ImageJ/Fiji Plugin for Calcium Transient Analysis
# A wizard-style interface to guide users through analyzing calcium dynamics
# to classify regenerating cells as cardiomyocytes or fibroblasts.
#
# Channel Assignment (based on fluorophores):
#   - Channel 1: Actin (RFP) -> Grays
#   - Channel 2: G0/G1 nuclei (CFP) -> Cyan  
#   - Channel 3: S/G2/M nuclei (miRFP670) -> Magenta
#   - Channel 4: Calcium (from time-series) -> Green
#
# Installation:
#   1. Copy this SINGLE file to Fiji.app/plugins/
#   2. Restart Fiji
#   3. Run via Plugins > CalciumAnalysis Wizard

from ij import IJ, ImagePlus, CompositeImage, WindowManager, ImageStack
from ij.gui import GenericDialog, WaitForUserDialog, MessageDialog, Plot
from ij.plugin.frame import RoiManager
from ij.io import OpenDialog, DirectoryChooser
from ij.process import FloatProcessor
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from java.awt import Color
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "bin_factor": 4,
    "n_magenta_rois": 5,   # S/G2/M regenerating cells (magenta nuclei)
    "n_cyan_rois": 5,      # G0/G1 quiescent cells (cyan nuclei)
    "n_bg_rois": 5,        # Background regions
    "baseline_frames": 50, # Frames for F0 calculation
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def bin_processor(ip, factor):
    """Bin a single processor by averaging pixels in factor x factor blocks."""
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
    
    return FloatProcessor(new_width, new_height, new_pixels)


def bin_image(imp, factor=4):
    """Bin an ImagePlus by the given factor using averaging."""
    original_width = imp.getWidth()
    original_height = imp.getHeight()
    new_width = original_width // factor
    new_height = original_height // factor
    n_channels = imp.getNChannels()
    n_slices = imp.getNSlices()
    n_frames = imp.getNFrames()
    
    out_stack = ImageStack(new_width, new_height)
    stack = imp.getStack()
    
    for i in range(1, stack.getSize() + 1):
        ip = stack.getProcessor(i).convertToFloat()
        binned_ip = bin_processor(ip, factor)
        out_stack.addSlice(stack.getSliceLabel(i), binned_ip)
    
    result = ImagePlus("Binned_" + imp.getTitle(), out_stack)
    
    if n_channels > 1 or n_slices > 1 or n_frames > 1:
        result.setDimensions(n_channels, n_slices, n_frames)
        if n_channels > 1:
            result = CompositeImage(result, CompositeImage.COMPOSITE)
    
    return result


def apply_composite_luts(imp, n_channels=4):
    """Apply LUTs: Grays (Actin), Cyan (G0/G1), Magenta (S/G2/M), Green (Calcium)."""
    if not isinstance(imp, CompositeImage):
        IJ.log("Warning: Image is not a CompositeImage")
        return
    
    imp.setDisplayMode(CompositeImage.COMPOSITE)
    imp.setC(1); IJ.run(imp, "Grays", "")    # Actin
    imp.setC(2); IJ.run(imp, "Cyan", "")     # G0/G1
    imp.setC(3); IJ.run(imp, "Magenta", "")  # S/G2/M
    if n_channels >= 4:
        imp.setC(4); IJ.run(imp, "Green", "")  # Calcium
    imp.setC(1)
    imp.updateAndDraw()


def auto_contrast(imp):
    """Apply auto-contrast to each channel."""
    n_channels = imp.getNChannels()
    for c in range(1, n_channels + 1):
        imp.setC(c)
        IJ.run(imp, "Enhance Contrast", "saturated=0.35")
    imp.setC(1)
    imp.updateAndDraw()


def get_roi_manager():
    """Get or create ROI Manager."""
    rm = RoiManager.getInstance()
    if rm is None:
        rm = RoiManager()
    return rm


def rename_rois(rm, prefix, count):
    """Rename the last 'count' ROIs with a prefix."""
    total_rois = rm.getCount()
    for i in range(count):
        roi_index = total_rois - count + i
        if roi_index >= 0:
            rm.rename(roi_index, "%s_%d" % (prefix, i + 1))


def save_rois(rm, filepath):
    """Save all ROIs to a ZIP file."""
    rm.runCommand("Deselect")
    rm.runCommand("Save", filepath)
    IJ.log("ROIs saved to: " + filepath)


def extract_traces(imp, rm):
    """Extract mean intensity traces from time-series for all ROIs."""
    n_frames = imp.getNFrames()
    if n_frames == 1:
        n_frames = imp.getNSlices()
    
    n_rois = rm.getCount()
    traces = {}
    roi_names = []
    
    for i in range(n_rois):
        name = rm.getName(i)
        roi_names.append(name)
        traces[name] = []
    
    IJ.log("Extracting traces for %d ROIs across %d frames..." % (n_rois, n_frames))
    
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
    """Calculate Delta F / F0 for all traces."""
    dff_traces = {}
    
    # Calculate background trace
    bg_trace = None
    if bg_rois:
        bg_values = [traces[name] for name in bg_rois if name in traces]
        if bg_values:
            n_frames = len(bg_values[0])
            bg_trace = [sum([bv[i] for bv in bg_values]) / len(bg_values) for i in range(n_frames)]
    
    for name, trace in traces.items():
        if bg_rois and name in bg_rois:
            continue
        
        if bg_trace:
            trace = [trace[i] - bg_trace[i] for i in range(len(trace))]
        
        f0 = sum(trace[:baseline_frames]) / baseline_frames if baseline_frames > 0 else 1.0
        if f0 <= 0:
            f0 = 1.0
        
        dff = [(f - f0) / f0 for f in trace]
        dff_traces[name] = dff
    
    return dff_traces


def detect_transients(trace, min_prominence=0.1, min_height=0.05):
    """Simple peak detection for calcium transients."""
    peaks = []
    n = len(trace)
    
    for i in range(1, n - 1):
        if trace[i] > trace[i-1] and trace[i] > trace[i+1] and trace[i] > min_height:
            left_min = min(trace[max(0, i-20):i]) if i > 0 else trace[0]
            right_min = min(trace[i+1:min(n, i+21)]) if i < n-1 else trace[-1]
            prominence = trace[i] - max(left_min, right_min)
            if prominence > min_prominence:
                peaks.append(i)
    
    return peaks


def plot_traces(dff_traces, title="Calcium Transients"):
    """Create a plot of all dF/F0 traces."""
    if not dff_traces:
        return None
    
    first_trace = list(dff_traces.values())[0]
    n_frames = len(first_trace)
    x_values = list(range(n_frames))
    
    plot = Plot(title, "Frame", "dF/F0")
    colors = {"Cyan": Color.CYAN, "Magenta": Color.MAGENTA, "BG": Color.GRAY}
    
    for name, trace in sorted(dff_traces.items()):
        color = Color.BLACK
        for prefix, c in colors.items():
            if name.startswith(prefix):
                color = c
                break
        plot.setColor(color)
        plot.addPoints(x_values, trace, Plot.LINE)
    
    plot.show()
    return plot


def export_traces_csv(dff_traces, filepath):
    """Export traces to CSV."""
    if not dff_traces:
        return
    
    roi_names = sorted(dff_traces.keys())
    n_frames = len(list(dff_traces.values())[0])
    
    with open(filepath, 'w') as f:
        f.write("Frame," + ",".join(roi_names) + "\n")
        for i in range(n_frames):
            row = [str(i)] + ["%.6f" % dff_traces[name][i] for name in roi_names]
            f.write(",".join(row) + "\n")
    
    IJ.log("Traces exported to: " + filepath)


def generate_summary(dff_traces, output_path):
    """Generate classification summary."""
    with open(output_path, 'w') as f:
        f.write("ROI,Type,NumPeaks,Classification\n")
        
        for name, trace in sorted(dff_traces.items()):
            if name.startswith("Magenta"):
                cell_type = "S/G2/M (Regenerating)"
            elif name.startswith("Cyan"):
                cell_type = "G0/G1 (Quiescent)"
            else:
                cell_type = "Unknown"
            
            peaks = detect_transients(trace)
            n_peaks = len(peaks)
            
            if n_peaks >= 3:
                classification = "Likely Cardiomyocyte"
            elif n_peaks >= 1:
                classification = "Possible Cardiomyocyte"
            else:
                classification = "Likely Fibroblast"
            
            f.write("%s,%s,%d,%s\n" % (name, cell_type, n_peaks, classification))
    
    IJ.log("Summary exported to: " + output_path)


# =============================================================================
# WIZARD STEPS
# =============================================================================

def step_file_selection():
    """Step 1: Select static and time-series files."""
    gd = GenericDialog("Calcium Analysis Wizard - Step 1: File Selection")
    gd.addMessage("Select the microscopy files for analysis:")
    gd.addMessage("")
    gd.addMessage("Static Image: 3-channel (Actin, G0/G1, S/G2/M)")
    gd.addMessage("Time Series: Calcium recording")
    gd.addMessage("")
    gd.addMessage("Click OK to select files...")
    
    gd.showDialog()
    if gd.wasCanceled():
        return None, None
    
    od_static = OpenDialog("Select Static 3-Channel Image (.nd2)", "")
    static_path = od_static.getPath()
    if static_path is None:
        return None, None
    
    od_ts = OpenDialog("Select Time-Series Calcium Recording (.nd2)", "")
    ts_path = od_ts.getPath()
    if ts_path is None:
        return None, None
    
    IJ.log("Static: " + static_path)
    IJ.log("Time-series: " + ts_path)
    return static_path, ts_path


def step_output_selection():
    """Select output directory."""
    dc = DirectoryChooser("Select Output Directory")
    return dc.getDirectory()


def step_load_and_merge(static_path, ts_path):
    """Step 2: Load images and create 4-channel merge."""
    IJ.log("Loading static image...")
    IJ.showStatus("Loading static image...")
    
    try:
        opts = ImporterOptions()
        opts.setId(static_path)
        opts.setOpenAllSeries(False)
        opts.setSeriesOn(0, True)
        imps = BF.openImagePlus(opts)
        if imps is None or len(imps) == 0:
            IJ.error("Failed to open static image")
            return None, None
        static_imp = imps[0]
        IJ.log("Static: %dx%d, %d ch" % (static_imp.getWidth(), static_imp.getHeight(), static_imp.getNChannels()))
    except Exception as e:
        IJ.error("Error: " + str(e))
        return None, None
    
    # Bin static image
    IJ.log("Binning static image %dx..." % CONFIG["bin_factor"])
    binned_static = bin_image(static_imp, CONFIG["bin_factor"])
    IJ.log("Binned: %dx%d" % (binned_static.getWidth(), binned_static.getHeight()))
    static_imp.close()
    
    # Load time-series
    IJ.log("Loading time-series...")
    IJ.showStatus("Loading time-series...")
    
    try:
        opts = ImporterOptions()
        opts.setId(ts_path)
        opts.setVirtual(True)
        opts.setOpenAllSeries(False)
        opts.setSeriesOn(0, True)
        imps = BF.openImagePlus(opts)
        if imps is None or len(imps) == 0:
            IJ.error("Failed to open time-series")
            return None, None
        ts_imp = imps[0]
        n_frames = ts_imp.getNFrames() if ts_imp.getNFrames() > 1 else ts_imp.getNSlices()
        IJ.log("Time-series: %dx%d, %d frames" % (ts_imp.getWidth(), ts_imp.getHeight(), n_frames))
    except Exception as e:
        IJ.error("Error: " + str(e))
        return None, None
    
    # Check dimensions
    if binned_static.getWidth() != ts_imp.getWidth() or binned_static.getHeight() != ts_imp.getHeight():
        IJ.error("Dimension mismatch! Binned static (%dx%d) vs time-series (%dx%d)" %
            (binned_static.getWidth(), binned_static.getHeight(), ts_imp.getWidth(), ts_imp.getHeight()))
        return None, None
    
    # Create 4-channel merge
    IJ.log("Creating 4-channel merge...")
    ts_imp.setSlice(1)
    calcium_ip = ts_imp.getProcessor().duplicate()
    
    width, height = binned_static.getWidth(), binned_static.getHeight()
    merged_stack = ImageStack(width, height)
    
    static_stack = binned_static.getStack()
    for c in range(1, 4):
        merged_stack.addSlice("C%d" % c, static_stack.getProcessor(c).duplicate())
    merged_stack.addSlice("Calcium", calcium_ip)
    
    merged_imp = ImagePlus("4-Channel_Merged", merged_stack)
    merged_imp.setDimensions(4, 1, 1)
    merged_composite = CompositeImage(merged_imp, CompositeImage.COMPOSITE)
    
    binned_static.close()
    IJ.log("4-channel merge created")
    return merged_composite, ts_imp


def step_setup_composite_view(merged_imp):
    """Step 3: Setup composite view with LUTs."""
    apply_composite_luts(merged_imp, n_channels=4)
    auto_contrast(merged_imp)
    merged_imp.show()
    merged_imp.setTitle("4Channel_Composite")
    
    IJ.run("Brightness/Contrast...")
    IJ.run("Channels Tool...")
    
    WaitForUserDialog("Step 3: Adjust Visualization",
        "Adjust brightness/contrast as needed.\n\n"
        "Channels:\n"
        "  Grays (Ch1): Actin\n"
        "  Cyan (Ch2): G0/G1 nuclei\n"
        "  Magenta (Ch3): S/G2/M nuclei\n"
        "  Green (Ch4): Calcium\n\n"
        "Click OK when ready.").show()
    
    return merged_imp


def step_roi_selection(imp, roi_type, prefix, count, instruction):
    """Generic ROI selection step."""
    rm = get_roi_manager()
    initial_count = rm.getCount()
    
    gd = GenericDialog("ROI Selection: %s" % roi_type)
    gd.addMessage(instruction)
    gd.addMessage("")
    gd.addMessage("1. Use Freehand/Polygon tool")
    gd.addMessage("2. Draw around each cell")
    gd.addMessage("3. Press 'T' to add to ROI Manager")
    gd.addMessage("4. Repeat for %d ROIs" % count)
    gd.addMessage("")
    gd.addMessage("Click OK to start.")
    
    gd.showDialog()
    if gd.wasCanceled():
        return False
    
    WindowManager.setCurrentWindow(imp.getWindow())
    imp.getWindow().toFront()
    IJ.setTool("freehand")
    rm.setVisible(True)
    rm.toFront()
    
    WaitForUserDialog("Select %d %s" % (count, roi_type),
        "Add %d %s to ROI Manager.\nClick OK when done." % (count, roi_type)).show()
    
    new_count = rm.getCount() - initial_count
    if new_count < count:
        WaitForUserDialog("Need More ROIs",
            "Added %d, need %d more." % (new_count, count - new_count)).show()
        new_count = rm.getCount() - initial_count
    
    rename_rois(rm, prefix, new_count)
    IJ.log("Added %d ROIs for %s" % (new_count, roi_type))
    return True


def step_select_magenta_cells(imp):
    """Step 4: Select S/G2/M cells (magenta nuclei)."""
    return step_roi_selection(imp, "S/G2/M Cells", "Magenta", CONFIG["n_magenta_rois"],
        "Select cells with MAGENTA nuclei (Ch3).\n"
        "TIP: In Channels Tool, turn OFF Cyan (Ch2) and Green (Ch4).")


def step_select_cyan_cells(imp):
    """Step 5: Select G0/G1 cells (cyan nuclei)."""
    return step_roi_selection(imp, "G0/G1 Cells", "Cyan", CONFIG["n_cyan_rois"],
        "Select cells with CYAN nuclei (Ch2).\n"
        "TIP: In Channels Tool, turn OFF Magenta (Ch3) and Green (Ch4).")


def step_select_background(imp):
    """Step 6: Select background regions."""
    return step_roi_selection(imp, "Background", "BG", CONFIG["n_bg_rois"],
        "Select BACKGROUND regions (no cells).")


def step_extract_and_analyze(ts_imp, output_dir):
    """Step 7: Extract and analyze traces."""
    rm = get_roi_manager()
    if rm.getCount() == 0:
        IJ.error("No ROIs found")
        return False
    
    IJ.log("Extracting traces...")
    traces = extract_traces(ts_imp, rm)
    
    bg_names = [name for name in traces.keys() if name.startswith("BG")]
    
    IJ.log("Calculating dF/F0...")
    dff_traces = calculate_dff(traces, CONFIG["baseline_frames"], bg_names)
    
    IJ.log("Plotting...")
    plot_traces(dff_traces, "Calcium Transients - dF/F0")
    
    IJ.log("Exporting...")
    export_traces_csv(dff_traces, os.path.join(output_dir, "calcium_traces.csv"))
    generate_summary(dff_traces, os.path.join(output_dir, "transient_summary.csv"))
    
    return True


def step_finalize(output_dir):
    """Step 8: Save and finish."""
    rm = get_roi_manager()
    save_rois(rm, os.path.join(output_dir, "calcium_rois.zip"))
    
    IJ.showMessage("Analysis Complete",
        "Files saved to:\n" + output_dir + "\n\n"
        "- calcium_rois.zip\n"
        "- calcium_traces.csv\n"
        "- transient_summary.csv")
    IJ.log("Done!")


# =============================================================================
# MAIN WIZARD
# =============================================================================

def run_wizard():
    """Main entry point."""
    IJ.log("=" * 50)
    IJ.log("Calcium Analysis Wizard")
    IJ.log("=" * 50)
    
    static_path, ts_path = step_file_selection()
    if static_path is None:
        return
    
    output_dir = step_output_selection()
    if output_dir is None:
        return
    
    merged_imp, ts_imp = step_load_and_merge(static_path, ts_path)
    if merged_imp is None:
        return
    
    composite_imp = step_setup_composite_view(merged_imp)
    
    if not step_select_magenta_cells(composite_imp):
        return
    if not step_select_cyan_cells(composite_imp):
        return
    if not step_select_background(composite_imp):
        return
    
    if not step_extract_and_analyze(ts_imp, output_dir):
        return
    
    step_finalize(output_dir)


# Run wizard
if __name__ == "__main__" or __name__ == "__builtin__":
    run_wizard()
