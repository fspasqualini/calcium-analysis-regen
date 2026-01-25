# CalciumAnalysis_Wizard_v1_Auto.py
# ImageJ/Fiji Plugin for Calcium Transient Analysis - V1 AUTO
# Automated nuclear segmentation for ROI selection.
#
# VERSION: v1-auto (automated ROI selection via nuclear segmentation)
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
#   3. Run via Plugins > CalciumAnalysis Wizard v1 Auto

from ij import IJ, ImagePlus, CompositeImage, WindowManager, ImageStack
from ij.gui import GenericDialog, WaitForUserDialog, MessageDialog, Plot, OvalRoi, ShapeRoi, Overlay
from ij.plugin.frame import RoiManager
from ij.io import OpenDialog, DirectoryChooser, FileSaver
from ij.process import FloatProcessor
from ij.measure import ResultsTable
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
from java.awt import Color
import os
import random
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "bin_factor": 4,
    "n_magenta_rois": 5,   # S/G2/M regenerating cells (magenta nuclei)
    "n_cyan_rois": 5,      # G0/G1 quiescent cells (cyan nuclei)
    "n_bg_rois": 5,        # Background regions
    # Preprocessing defaults (applied to static image only)
    "preprocess_enabled": True,
    "bg_subtract_method": "rolling_ball",  # "none", "rolling_ball", "mosaic"
    "bg_subtract_radius": 50,              # Rolling ball radius in pixels
    "denoise_method": "median",            # "none", "median", "gaussian"
    "denoise_radius": 2,                   # Median radius or Gaussian sigma
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


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def check_mosaic_available():
    """Check if MOSAIC Background Subtractor is available."""
    try:
        # Try to access the MOSAIC plugin class
        from mosaic.plugins import BackgroundSubtractor
        return True
    except:
        return False


def apply_background_subtraction(imp, method, radius):
    """Apply background subtraction to all channels of an image."""
    if method == "none":
        return imp
    
    IJ.log("Applying background subtraction: %s (radius=%d)..." % (method, radius))
    
    # Work on a duplicate to preserve original
    result = imp.duplicate()
    
    if method == "rolling_ball":
        # Apply rolling ball to each channel
        n_channels = result.getNChannels()
        for c in range(1, n_channels + 1):
            result.setC(c)
            IJ.run(result, "Subtract Background...", 
                   "rolling=%d sliding" % radius)
        result.setC(1)
    
    elif method == "mosaic":
        # MOSAIC Background Subtractor
        if check_mosaic_available():
            n_channels = result.getNChannels()
            for c in range(1, n_channels + 1):
                result.setC(c)
                IJ.run(result, "Background Subtractor", "length=%d" % radius)
            result.setC(1)
        else:
            IJ.log("Warning: MOSAIC plugin not installed, falling back to rolling ball")
            return apply_background_subtraction(imp, "rolling_ball", radius)
    
    IJ.log("Background subtraction complete")
    return result


def apply_denoising(imp, method, radius):
    """Apply denoising filter to all channels of an image."""
    if method == "none":
        return imp
    
    IJ.log("Applying denoising: %s (radius=%.1f)..." % (method, radius))
    
    # Work on a duplicate to preserve original
    result = imp.duplicate()
    
    if method == "median":
        n_channels = result.getNChannels()
        for c in range(1, n_channels + 1):
            result.setC(c)
            IJ.run(result, "Median...", "radius=%d" % int(radius))
        result.setC(1)
    
    elif method == "gaussian":
        n_channels = result.getNChannels()
        for c in range(1, n_channels + 1):
            result.setC(c)
            IJ.run(result, "Gaussian Blur...", "sigma=%.1f" % radius)
        result.setC(1)
    
    IJ.log("Denoising complete")
    return result


def preprocess_image(imp, bg_method, bg_radius, denoise_method, denoise_radius):
    """Apply full preprocessing pipeline to an image."""
    result = imp.duplicate()
    
    # Step 1: Background subtraction
    if bg_method != "none":
        temp = apply_background_subtraction(result, bg_method, bg_radius)
        result.close()
        result = temp
    
    # Step 2: Denoising
    if denoise_method != "none":
        temp = apply_denoising(result, denoise_method, denoise_radius)
        result.close()
        result = temp
    
    return result


def show_preprocessing_preview(original_imp, processed_imp):
    """Show side-by-side preview of original and processed images."""
    # Create a montage for comparison
    original_imp.setTitle("ORIGINAL (left)")
    processed_imp.setTitle("PROCESSED (right)")
    
    # Apply LUTs to both for consistent viewing
    apply_composite_luts(original_imp, n_channels=3)
    apply_composite_luts(processed_imp, n_channels=3)
    auto_contrast(original_imp)
    auto_contrast(processed_imp)
    
    # Show both side by side
    original_imp.show()
    processed_imp.show()
    
    # Tile windows horizontally
    IJ.run("Tile")


def step_preprocessing_options():
    """Show preprocessing options dialog and return settings."""
    mosaic_available = check_mosaic_available()
    
    gd = GenericDialog("Step 2: Image Preprocessing")
    gd.addMessage("Preprocessing reduces background fluorescence and noise")
    gd.addMessage("to make cell detection easier.")
    gd.addMessage("")
    
    # Background subtraction options
    bg_methods = ["none", "rolling_ball"]
    if mosaic_available:
        bg_methods.append("mosaic")
    else:
        gd.addMessage("(MOSAIC plugin not installed - install via Help > Update...)")
    
    gd.addChoice("Background subtraction:", bg_methods, CONFIG["bg_subtract_method"])
    gd.addNumericField("  Radius (pixels):", CONFIG["bg_subtract_radius"], 0)
    
    gd.addMessage("")
    
    # Denoising options
    denoise_methods = ["none", "median", "gaussian"]
    gd.addChoice("Denoising:", denoise_methods, CONFIG["denoise_method"])
    gd.addNumericField("  Radius/Sigma:", CONFIG["denoise_radius"], 1)
    
    gd.addMessage("")
    gd.addCheckbox("Show preview before continuing", True)
    
    gd.showDialog()
    if gd.wasCanceled():
        return None
    
    settings = {
        "bg_method": gd.getNextChoice(),
        "bg_radius": int(gd.getNextNumber()),
        "denoise_method": gd.getNextChoice(),
        "denoise_radius": gd.getNextNumber(),
        "show_preview": gd.getNextBoolean()
    }
    
    return settings


def step_preprocessing_with_preview(binned_static):
    """Run preprocessing with preview and allow user to adjust."""
    while True:
        # Get preprocessing settings
        settings = step_preprocessing_options()
        if settings is None:
            return None  # User cancelled
        
        # Check if any preprocessing is enabled
        if settings["bg_method"] == "none" and settings["denoise_method"] == "none":
            IJ.log("No preprocessing selected, continuing...")
            return binned_static.duplicate()
        
        # Apply preprocessing
        IJ.log("Applying preprocessing...")
        processed = preprocess_image(
            binned_static,
            settings["bg_method"],
            settings["bg_radius"],
            settings["denoise_method"],
            settings["denoise_radius"]
        )
        
        if settings["show_preview"]:
            # Show preview
            preview_original = binned_static.duplicate()
            show_preprocessing_preview(preview_original, processed)
            
            # Ask user if satisfied
            gd = GenericDialog("Preprocessing Preview")
            gd.addMessage("Compare ORIGINAL (left) vs PROCESSED (right)")
            gd.addMessage("")
            gd.addMessage("Click OK to accept, or Cancel to adjust settings.")
            gd.showDialog()
            
            # Close preview windows
            preview_original.close()
            
            if gd.wasCanceled():
                processed.close()
                continue  # Loop back to settings
            else:
                return processed
        else:
            return processed


# =============================================================================
# AUTOMATED SEGMENTATION FUNCTIONS
# =============================================================================

def segment_nuclei(channel_ip, original_ip, min_size=30, max_size=500):
    """
    Segment nuclei from a channel using Otsu threshold + Analyze Particles.
    Returns a list of ROIs sorted by mean intensity (brightest first).
    
    Args:
        channel_ip: ImageProcessor for thresholding
        original_ip: Original channel ImageProcessor for intensity measurement
        min_size: Minimum nucleus area in pixels (default 30 for ~6px diameter)
        max_size: Maximum nucleus area in pixels (default 500 for ~25px diameter)
    """
    # Create a temporary image for processing
    temp_imp = ImagePlus("temp_segment", channel_ip.duplicate())
    
    # Apply Otsu auto-threshold
    IJ.setAutoThreshold(temp_imp, "Otsu dark")
    IJ.run(temp_imp, "Convert to Mask", "")
    
    # Clean up the mask - more aggressive erosion/dilation to separate touching nuclei
    IJ.run(temp_imp, "Erode", "")
    IJ.run(temp_imp, "Erode", "")
    IJ.run(temp_imp, "Dilate", "")
    IJ.run(temp_imp, "Dilate", "")
    IJ.run(temp_imp, "Fill Holes", "")
    IJ.run(temp_imp, "Watershed", "")
    
    # Analyze Particles to get ROIs - tighter circularity for round nuclei
    IJ.run("Set Measurements...", "area mean centroid shape redirect=None decimal=3")
    
    # Get ROI Manager and clear it temporarily
    rm = get_roi_manager()
    initial_count = rm.getCount()
    
    # Run Analyze Particles with stricter circularity (0.5-1.0 for round objects)
    IJ.run(temp_imp, "Analyze Particles...", 
           "size=%d-%d circularity=0.5-1.00 show=Nothing add" % (min_size, max_size))
    
    # Collect new ROIs with intensity from ORIGINAL image
    new_count = rm.getCount()
    rois_with_intensity = []
    
    # Create temp image from original for intensity measurement
    orig_imp = ImagePlus("orig_measure", original_ip.duplicate())
    
    for i in range(initial_count, new_count):
        roi = rm.getRoi(i)
        # Measure mean intensity on original (not thresholded) image
        orig_imp.setRoi(roi)
        stats = orig_imp.getProcessor().getStatistics()
        mean_intensity = stats.mean
        area = stats.area
        rois_with_intensity.append((roi.clone(), mean_intensity, area))
    
    orig_imp.close()
    
    # Remove temporary ROIs from manager
    for i in range(new_count - 1, initial_count - 1, -1):
        rm.select(i)
        rm.runCommand("Delete")
    
    temp_imp.close()
    
    # Sort by INTENSITY (brightest first) - these are the nuclei in focus
    rois_with_intensity.sort(key=lambda x: x[1], reverse=True)
    
    # Return as (roi, area) tuples for compatibility
    return [(roi, area) for roi, intensity, area in rois_with_intensity]


def select_top_rois(rois_with_area, n=5):
    """Select the top n ROIs from a pre-sorted list of (roi, area) tuples."""
    return [roi for roi, area in rois_with_area[:n]]


def get_calcium_footprint_mask(calcium_ip, threshold_percentile=10):
    """
    Create a mask of the calcium signal footprint.
    Returns a binary mask where calcium signal is present.
    """
    temp_imp = ImagePlus("calcium_mask", calcium_ip.duplicate())
    
    # Get threshold value at given percentile
    stats = temp_imp.getProcessor().getStatistics()
    threshold = stats.min + (stats.max - stats.min) * (threshold_percentile / 100.0)
    
    # Apply threshold
    temp_imp.getProcessor().setThreshold(threshold, stats.max, 0)
    IJ.run(temp_imp, "Convert to Mask", "")
    
    # Dilate to fill gaps
    IJ.run(temp_imp, "Dilate", "")
    IJ.run(temp_imp, "Dilate", "")
    IJ.run(temp_imp, "Fill Holes", "")
    
    return temp_imp


def get_organoid_mask(actin_ip, threshold_percentile=30):
    """
    Create a mask of the organoid region (high actin signal).
    Returns a binary mask where organoid is present.
    """
    temp_imp = ImagePlus("organoid_mask", actin_ip.duplicate())
    
    # Get threshold value at given percentile
    stats = temp_imp.getProcessor().getStatistics()
    threshold = stats.min + (stats.max - stats.min) * (threshold_percentile / 100.0)
    
    # Apply threshold
    temp_imp.getProcessor().setThreshold(threshold, stats.max, 0)
    IJ.run(temp_imp, "Convert to Mask", "")
    
    # Dilate to expand organoid boundary
    for _ in range(5):
        IJ.run(temp_imp, "Dilate", "")
    IJ.run(temp_imp, "Fill Holes", "")
    
    return temp_imp


def create_organoid_convex_hull(cyan_ip):
    """
    Create a convex hull mask from the cyan nuclear channel.
    Returns the convex hull mask image and its expanded version (10% dilation).
    """
    # Create threshold mask
    temp_imp = ImagePlus("cyan_for_hull", cyan_ip.duplicate())
    IJ.setAutoThreshold(temp_imp, "Otsu dark")
    IJ.run(temp_imp, "Convert to Mask", "")
    
    # Clean up
    IJ.run(temp_imp, "Fill Holes", "")
    
    # Get convex hull using Selection > Convex Hull
    # First, create a selection from the mask
    IJ.run(temp_imp, "Create Selection", "")
    roi = temp_imp.getRoi()
    
    if roi is None:
        IJ.log("Warning: Could not create selection for convex hull")
        temp_imp.close()
        return None, None
    
    # Get convex hull
    from ij.gui import PolygonRoi, Roi
    if hasattr(roi, 'getConvexHull'):
        hull_polygon = roi.getConvexHull()
        if hull_polygon:
            hull_roi = PolygonRoi(hull_polygon, Roi.POLYGON)
        else:
            hull_roi = roi
    else:
        # Fallback - use the selection as-is
        hull_roi = roi
    
    # Create mask from convex hull
    width = temp_imp.getWidth()
    height = temp_imp.getHeight()
    
    hull_imp = ImagePlus("hull_mask", temp_imp.getProcessor().createProcessor(width, height))
    hull_imp.getProcessor().setColor(255)
    hull_imp.setRoi(hull_roi)
    # Use fill instead of fillRoi for compatibility
    IJ.run(hull_imp, "Fill", "slice")
    hull_imp.killRoi()
    
    # Create expanded version (10% dilation)
    expanded_imp = hull_imp.duplicate()
    expanded_imp.setTitle("expanded_hull")
    
    # Calculate number of dilations for ~10% expansion
    # Approximate: if avg radius is R, 10% = 0.1R. For a 300px wide organoid, ~15 dilations
    n_dilations = max(5, width // 40)  # At least 5, scales with image size
    IJ.log("  Dilating hull %d times for ~10%% expansion" % n_dilations)
    
    for _ in range(n_dilations):
        IJ.run(expanded_imp, "Dilate", "")
    
    temp_imp.close()
    return hull_imp, expanded_imp


def place_background_rois(cyan_ip, avg_nucleus_area, n_rois=5):
    """
    Place circular background ROIs just outside the organoid convex hull.
    Uses cyan nuclear channel to define organoid boundary.
    """
    width = cyan_ip.getWidth()
    height = cyan_ip.getHeight()
    
    # Create convex hull and expanded version
    IJ.log("  Creating convex hull from cyan nuclei...")
    hull_imp, expanded_imp = create_organoid_convex_hull(cyan_ip)
    
    if hull_imp is None or expanded_imp is None:
        IJ.log("Warning: Could not create convex hull, using corners")
        radius = max(5, int(math.sqrt(avg_nucleus_area / math.pi)))
        margin = radius + 20
        corners = [
            (margin, margin),
            (width - margin, margin),
            (margin, height - margin),
            (width - margin, height - margin),
            (width // 2, margin),
        ]
        rois = []
        for x, y in corners[:n_rois]:
            roi = OvalRoi(x - radius, y - radius, radius * 2, radius * 2)
            rois.append(roi)
        return rois
    
    hull_ip = hull_imp.getProcessor()
    expanded_ip = expanded_imp.getProcessor()
    
    # Calculate radius from average nucleus area
    radius = max(5, int(math.sqrt(avg_nucleus_area / math.pi)))
    IJ.log("  Background ROI radius: %d pixels" % radius)
    
    # Find positions OUTSIDE expanded hull (in the ring around it)
    # but not too far from the organoid (within image bounds with margin)
    valid_positions = []
    step = max(radius // 2, 5)
    margin = radius + 10
    
    for y in range(margin, height - margin, step):
        for x in range(margin, width - margin, step):
            # Get pixel values
            expanded_val = expanded_ip.getPixel(x, y)
            
            # Position should be OUTSIDE the expanded hull (black = 0)
            if expanded_val < 128:
                # But check that we can see some hull nearby (within 50px)
                # This ensures we're not in far corners
                valid_positions.append((x, y))
    
    IJ.log("  Found %d positions outside expanded hull" % len(valid_positions))
    
    # Clean up
    hull_imp.close()
    expanded_imp.close()
    
    if len(valid_positions) < n_rois:
        IJ.log("Warning: Only found %d valid background positions (need %d)" % 
               (len(valid_positions), n_rois))
        # Fallback to corners
        if len(valid_positions) == 0:
            margin = radius + 20
            corners = [
                (margin, margin),
                (width - margin, margin),
                (margin, height - margin),
                (width - margin, height - margin),
                (width // 2, margin),
            ]
            valid_positions = corners[:n_rois]
    
    # Randomly select positions
    if len(valid_positions) >= n_rois:
        selected_positions = random.sample(valid_positions, n_rois)
    else:
        selected_positions = valid_positions
    
    # Create circular ROIs
    rois = []
    for x, y in selected_positions:
        roi = OvalRoi(x - radius, y - radius, radius * 2, radius * 2)
        rois.append(roi)
    
    return rois


def auto_segment_all_rois(merged_imp, ts_imp_systole_frame):
    """
    Automatically segment nuclei and place background ROIs.
    Returns (cyan_rois, magenta_rois, bg_rois, avg_nucleus_area).
    """
    IJ.log("=== AUTO-SEGMENTATION ===")
    
    # Get channel processors from the merged image (use first timepoint)
    merged_imp.setT(1)  # Diastole frame
    
    # Extract channel images
    merged_imp.setC(1)  # Actin
    actin_ip = merged_imp.getProcessor().duplicate()
    
    merged_imp.setC(2)  # Cyan (G0/G1)
    cyan_ip = merged_imp.getProcessor().duplicate()
    
    merged_imp.setC(3)  # Magenta (S/G2/M)
    magenta_ip = merged_imp.getProcessor().duplicate()
    
    merged_imp.setC(4)  # Calcium
    merged_imp.setT(2)  # Systole frame for calcium footprint
    calcium_ip = merged_imp.getProcessor().duplicate()
    merged_imp.setT(1)  # Reset
    merged_imp.setC(1)  # Reset
    
    # Segment Cyan nuclei (pass same ip for threshold and measurement)
    IJ.log("Segmenting Cyan (G0/G1) nuclei...")
    cyan_rois_with_area = segment_nuclei(cyan_ip, cyan_ip)
    IJ.log("  Found %d Cyan nuclei" % len(cyan_rois_with_area))
    
    # Segment Magenta nuclei
    IJ.log("Segmenting Magenta (S/G2/M) nuclei...")
    magenta_rois_with_area = segment_nuclei(magenta_ip, magenta_ip)
    IJ.log("  Found %d Magenta nuclei" % len(magenta_rois_with_area))
    
    # Select top 5 brightest of each
    n_cyan = min(5, len(cyan_rois_with_area))
    n_magenta = min(5, len(magenta_rois_with_area))
    
    if n_cyan < 5:
        IJ.log("  WARNING: Only %d Cyan nuclei found (wanted 5)" % n_cyan)
    if n_magenta < 5:
        IJ.log("  WARNING: Only %d Magenta nuclei found (wanted 5)" % n_magenta)
    
    cyan_rois = select_top_rois(cyan_rois_with_area, n_cyan)
    magenta_rois = select_top_rois(magenta_rois_with_area, n_magenta)
    
    # Calculate average nucleus area for background ROI sizing
    all_areas = [area for _, area in cyan_rois_with_area[:n_cyan]] + \
                [area for _, area in magenta_rois_with_area[:n_magenta]]
    avg_nucleus_area = sum(all_areas) / len(all_areas) if all_areas else 100
    IJ.log("  Average nucleus area: %.1f pixels" % avg_nucleus_area)
    
    # Place background ROIs using convex hull of cyan nuclei
    IJ.log("Placing background ROIs (convex hull method)...")
    bg_rois = place_background_rois(cyan_ip, avg_nucleus_area, n_rois=5)
    IJ.log("  Placed %d background ROIs" % len(bg_rois))
    
    return cyan_rois, magenta_rois, bg_rois, avg_nucleus_area


def add_rois_to_manager(rois, prefix, rm):
    """Add a list of ROIs to the ROI Manager with numbered names."""
    for i, roi in enumerate(rois):
        roi.setName("%s_%d" % (prefix, i + 1))
        rm.addRoi(roi)


def save_roi_verification_image(merged_imp, rm, output_dir):
    """Save an image with ROIs overlaid for verification."""
    # Create a flattened RGB image with ROIs
    merged_imp.setT(2)  # Show systole (high calcium) for verification
    merged_imp.setDisplayMode(CompositeImage.COMPOSITE)
    
    # Create overlay from ROI Manager
    overlay = Overlay()
    colors = {"Cyan": Color.CYAN, "Magenta": Color.MAGENTA, "BG": Color.YELLOW}
    
    for i in range(rm.getCount()):
        roi = rm.getRoi(i)
        name = rm.getName(i)
        roi_copy = roi.clone()
        
        # Set color based on type
        for prefix, color in colors.items():
            if name.startswith(prefix):
                roi_copy.setStrokeColor(color)
                roi_copy.setStrokeWidth(2)
                break
        
        overlay.add(roi_copy)
    
    merged_imp.setOverlay(overlay)
    
    # Flatten and save
    flat_imp = merged_imp.flatten()
    flat_imp.setTitle("ROI_Verification")
    
    save_path = os.path.join(output_dir, "roi_verification.png")
    FileSaver(flat_imp).saveAsPng(save_path)
    IJ.log("ROI verification image saved to: " + save_path)
    
    # Show the image for user review
    flat_imp.show()
    
    # Remove overlay from original
    merged_imp.setOverlay(None)
    merged_imp.setT(1)
    
    return flat_imp


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


def calculate_dff(traces, bg_rois=None):
    """Calculate Delta F / F0 for all traces, including BG traces."""
    dff_traces = {}
    
    # Calculate average background trace (for subtraction from cell traces)
    bg_trace = None
    if bg_rois:
        bg_values = [traces[name] for name in bg_rois if name in traces]
        if bg_values:
            n_frames = len(bg_values[0])
            bg_trace = [sum([bv[i] for bv in bg_values]) / len(bg_values) for i in range(n_frames)]
    
    for name, trace in traces.items():
        is_bg = bg_rois and name in bg_rois
        
        # For BG traces: don't subtract background, just calculate dF/F0
        # For cell traces: subtract background first, then calculate dF/F0
        if is_bg:
            # BG traces: just calculate dF/F0 directly (no background subtraction)
            working_trace = trace
        else:
            # Cell traces: subtract average background
            if bg_trace:
                working_trace = [trace[i] - bg_trace[i] for i in range(len(trace))]
            else:
                working_trace = trace
        
        f0 = min(working_trace)  # Use minimum as baseline for spontaneously beating preparations
        if f0 <= 0:
            f0 = 1.0
        
        ff0 = [f / f0 for f in working_trace]
        dff_traces[name] = ff0
    
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


def calculate_group_stats(traces_list):
    """Calculate mean, std, min, max for a group of traces."""
    if not traces_list:
        return None, None, None, None
    
    n_frames = len(traces_list[0])
    n_traces = len(traces_list)
    
    mean_trace = []
    std_trace = []
    min_trace = []
    max_trace = []
    
    for i in range(n_frames):
        values = [traces_list[j][i] for j in range(n_traces)]
        mean_val = sum(values) / n_traces
        mean_trace.append(mean_val)
        
        # Standard deviation
        if n_traces > 1:
            variance = sum((v - mean_val) ** 2 for v in values) / (n_traces - 1)
            std_trace.append(variance ** 0.5)
        else:
            std_trace.append(0.0)
        
        min_trace.append(min(values))
        max_trace.append(max(values))
    
    return mean_trace, std_trace, min_trace, max_trace


def group_traces_by_type(traces_dict):
    """Group traces by cell type (Magenta, Cyan, BG)."""
    groups = {
        "Magenta": {"names": [], "traces": []},
        "Cyan": {"names": [], "traces": []},
        "BG": {"names": [], "traces": []}
    }
    
    for name, trace in traces_dict.items():
        if name.startswith("Magenta"):
            groups["Magenta"]["names"].append(name)
            groups["Magenta"]["traces"].append(trace)
        elif name.startswith("Cyan"):
            groups["Cyan"]["names"].append(name)
            groups["Cyan"]["traces"].append(trace)
        elif name.startswith("BG"):
            groups["BG"]["names"].append(name)
            groups["BG"]["traces"].append(trace)
    
    return groups


def plot_group_panel(group_name, traces_list, title_suffix, color, light_color):
    """Create a plot for a single group with individual traces + mean + envelope."""
    if not traces_list:
        return None
    
    n_frames = len(traces_list[0])
    x_values = list(range(n_frames))
    
    plot = Plot("%s - %s" % (group_name, title_suffix), "Frame", "F/F0")
    
    # Draw individual traces (lighter color)
    plot.setColor(light_color)
    plot.setLineWidth(1)
    for trace in traces_list:
        plot.addPoints(x_values, trace, Plot.LINE)
    
    # Calculate and draw mean + envelope
    mean_trace, std_trace, min_trace, max_trace = calculate_group_stats(traces_list)
    
    # Draw envelope (min/max as dashed lines)
    plot.setColor(Color.LIGHT_GRAY)
    plot.setLineWidth(1)
    plot.addPoints(x_values, min_trace, Plot.LINE)
    plot.addPoints(x_values, max_trace, Plot.LINE)
    
    # Draw mean as bold line
    plot.setColor(color)
    plot.setLineWidth(3)
    plot.addPoints(x_values, mean_trace, Plot.LINE)
    
    # Add legend
    plot.setColor(Color.BLACK)
    plot.addLabel(0.02, 0.98, "n=%d, bold=mean, gray=min/max" % len(traces_list))
    
    plot.show()
    return plot


def plot_grouped_traces(dff_traces, title_suffix="dF/F0"):
    """Create separate panel plots for each cell type group."""
    if not dff_traces:
        return []
    
    groups = group_traces_by_type(dff_traces)
    plots = []
    
    # Define colors for each group
    group_configs = {
        "Magenta": {"title": "S/G2/M Regenerating", "color": Color.MAGENTA, 
                    "light": Color(255, 200, 255)},
        "Cyan": {"title": "G0/G1 Quiescent", "color": Color.CYAN,
                 "light": Color(200, 255, 255)},
        "BG": {"title": "Background", "color": Color.GRAY,
               "light": Color(220, 220, 220)}
    }
    
    for group_key, config in group_configs.items():
        traces_list = groups[group_key]["traces"]
        if traces_list:
            plot = plot_group_panel(
                config["title"], 
                traces_list, 
                title_suffix,
                config["color"],
                config["light"]
            )
            if plot:
                plots.append(plot)
    
    return plots


def plot_traces(dff_traces, title="Calcium Transients"):
    """Create grouped panel plots: Cyan, Magenta, and Background (no 'All Traces')."""
    # Create grouped panel plots - this creates 3 panels: Cyan, Magenta, BG
    plots = plot_grouped_traces(dff_traces, title)
    
    if not plots:
        IJ.log("Warning: No plots created")
    
    return plots[0] if plots else None


def export_traces_csv(raw_traces, dff_traces, output_dir):
    """Export both raw and normalized traces to separate CSV files with group stats."""
    if not raw_traces or not dff_traces:
        return
    
    # Get frame count
    first_trace = list(raw_traces.values())[0]
    n_frames = len(first_trace)
    
    # Group traces
    raw_groups = group_traces_by_type(raw_traces)
    dff_groups = group_traces_by_type(dff_traces)
    
    # === Export RAW traces ===
    raw_filepath = os.path.join(output_dir, "calcium_traces_raw.csv")
    roi_names = sorted(raw_traces.keys())
    
    with open(raw_filepath, 'w') as f:
        f.write("Frame," + ",".join(roi_names) + "\n")
        for i in range(n_frames):
            row = [str(i)] + ["%.2f" % raw_traces[name][i] for name in roi_names]
            f.write(",".join(row) + "\n")
    
    IJ.log("Raw traces exported to: " + raw_filepath)
    
    # === Export dF/F0 traces with group statistics ===
    dff_filepath = os.path.join(output_dir, "calcium_traces_dff.csv")
    dff_names = sorted(dff_traces.keys())
    
    # Calculate group stats
    group_stats = {}
    for group_key in ["Magenta", "Cyan", "BG"]:
        traces_list = dff_groups[group_key]["traces"]
        if traces_list:
            mean_t, std_t, min_t, max_t = calculate_group_stats(traces_list)
            group_stats[group_key] = {"mean": mean_t, "std": std_t, "min": min_t, "max": max_t}
    
    # Build header: individual ROIs + group stats
    header_parts = ["Frame"] + dff_names
    for group_key in ["Magenta", "Cyan", "BG"]:
        if group_key in group_stats:
            header_parts.extend([
                "%s_mean" % group_key,
                "%s_std" % group_key,
                "%s_min" % group_key,
                "%s_max" % group_key
            ])
    
    with open(dff_filepath, 'w') as f:
        f.write(",".join(header_parts) + "\n")
        for i in range(n_frames):
            row = [str(i)]
            # Individual traces
            row.extend(["%.6f" % dff_traces[name][i] for name in dff_names])
            # Group stats
            for group_key in ["Magenta", "Cyan", "BG"]:
                if group_key in group_stats:
                    stats = group_stats[group_key]
                    row.extend([
                        "%.6f" % stats["mean"][i],
                        "%.6f" % stats["std"][i],
                        "%.6f" % stats["min"][i],
                        "%.6f" % stats["max"][i]
                    ])
            f.write(",".join(row) + "\n")
    
    IJ.log("dF/F0 traces exported to: " + dff_filepath)


def generate_summary(dff_traces, output_path):
    """Generate classification summary with group statistics."""
    # Calculate per-group transient counts
    groups = group_traces_by_type(dff_traces)
    
    with open(output_path, 'w') as f:
        f.write("ROI,Type,NumPeaks,Classification\n")
        
        group_peak_counts = {"Magenta": [], "Cyan": [], "BG": []}
        
        for name, trace in sorted(dff_traces.items()):
            if name.startswith("Magenta"):
                cell_type = "S/G2/M (Regenerating)"
                group_key = "Magenta"
            elif name.startswith("Cyan"):
                cell_type = "G0/G1 (Quiescent)"
                group_key = "Cyan"
            else:
                cell_type = "Background"
                group_key = "BG"
            
            peaks = detect_transients(trace)
            n_peaks = len(peaks)
            group_peak_counts[group_key].append(n_peaks)
            
            if name.startswith("BG"):
                classification = "N/A"
            elif n_peaks >= 3:
                classification = "Likely Cardiomyocyte"
            elif n_peaks >= 1:
                classification = "Possible Cardiomyocyte"
            else:
                classification = "Likely Fibroblast"
            
            f.write("%s,%s,%d,%s\n" % (name, cell_type, n_peaks, classification))
        
        # Add group summary at the end
        f.write("\n# Group Summary\n")
        for group_key, counts in group_peak_counts.items():
            if counts:
                avg_peaks = sum(counts) / len(counts)
                f.write("# %s: n=%d, avg_peaks=%.1f\n" % (group_key, len(counts), avg_peaks))
    
    IJ.log("Summary exported to: " + output_path)


# =============================================================================
# WIZARD STEPS
# =============================================================================

def step_file_selection():
    """Step 1: Select static and time-series files."""
    # Welcome dialog
    gd = GenericDialog("Calcium Analysis Wizard - Step 1: File Selection")
    gd.addMessage("This wizard will guide you through calcium transient analysis.")
    gd.addMessage("")
    gd.addMessage("You will need to select 3 things in sequence:")
    gd.addMessage("  1. HIGH-RES static image (cell structure + nuclei)")
    gd.addMessage("  2. LOW-RES calcium time-series video")
    gd.addMessage("  3. Output folder to save results")
    gd.addMessage("")
    gd.addMessage("Click OK to begin.")
    gd.showDialog()
    if gd.wasCanceled():
        return None, None
    
    # Pre-dialog for static image
    gd_static = GenericDialog("Step 1a: Static Image")
    gd_static.addMessage("In the NEXT window, select the:")
    gd_static.addMessage("")
    gd_static.addMessage("  >> HIGH-RESOLUTION CELL & NUCLEI IMAGE <<")
    gd_static.addMessage("")
    gd_static.addMessage("This is the 3-channel static image containing:")
    gd_static.addMessage("  - Actin (cell structure)")
    gd_static.addMessage("  - G0/G1 nuclei (quiescent cells)")
    gd_static.addMessage("  - S/G2/M nuclei (regenerating cells)")
    gd_static.addMessage("")
    gd_static.addMessage("Typically 2720x2720 pixels, .nd2 or .tif format")
    gd_static.showDialog()
    if gd_static.wasCanceled():
        return None, None
    
    od_static = OpenDialog("Select HIGH-RES Static Image (cell + nuclei)", "")
    static_path = od_static.getPath()
    if static_path is None:
        return None, None
    
    # Pre-dialog for time-series
    gd_ts = GenericDialog("Step 1b: Calcium Video")
    gd_ts.addMessage("In the NEXT window, select the:")
    gd_ts.addMessage("")
    gd_ts.addMessage("  >> LOW-RESOLUTION CALCIUM TIME-SERIES VIDEO <<")
    gd_ts.addMessage("")
    gd_ts.addMessage("This is the calcium recording containing:")
    gd_ts.addMessage("  - Single channel (calcium indicator)")
    gd_ts.addMessage("  - Many frames over time (~1000 frames)")
    gd_ts.addMessage("")
    gd_ts.addMessage("Typically 680x680 pixels (4x binned), .nd2 or .tif format")
    gd_ts.showDialog()
    if gd_ts.wasCanceled():
        return None, None
    
    od_ts = OpenDialog("Select LOW-RES Calcium Time-Series Video", "")
    ts_path = od_ts.getPath()
    if ts_path is None:
        return None, None
    
    IJ.log("Static: " + static_path)
    IJ.log("Time-series: " + ts_path)
    return static_path, ts_path


def step_output_selection():
    """Select output directory."""
    # Pre-dialog for output folder
    gd_out = GenericDialog("Step 1c: Output Folder")
    gd_out.addMessage("In the NEXT window, select the:")
    gd_out.addMessage("")
    gd_out.addMessage("  >> FOLDER TO SAVE RESULTS <<")
    gd_out.addMessage("")
    gd_out.addMessage("The wizard will save these files there:")
    gd_out.addMessage("  - calcium_rois.zip (ROI definitions)")
    gd_out.addMessage("  - calcium_traces.csv (intensity traces)")
    gd_out.addMessage("  - transient_summary.csv (cell classification)")
    gd_out.showDialog()
    if gd_out.wasCanceled():
        return None
    
    dc = DirectoryChooser("Select OUTPUT FOLDER for Results")
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
    
    # === NEW: Preprocessing step ===
    IJ.log("Starting preprocessing...")
    preprocessed_static = step_preprocessing_with_preview(binned_static)
    if preprocessed_static is None:
        binned_static.close()
        return None, None
    
    # Close original binned if preprocessing created a new image
    if preprocessed_static != binned_static:
        binned_static.close()
    binned_static = preprocessed_static
    
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
    
    # === Find global min (diastole) and max (systole) frames ===
    IJ.log("Finding diastole (min) and systole (max) frames...")
    IJ.showStatus("Analyzing calcium time-series...")
    
    n_frames = ts_imp.getNFrames() if ts_imp.getNFrames() > 1 else ts_imp.getNSlices()
    min_intensity = float('inf')
    max_intensity = float('-inf')
    min_frame = 1
    max_frame = 1
    
    for frame in range(1, n_frames + 1):
        if frame % 100 == 0:
            IJ.showProgress(frame, n_frames)
        ts_imp.setSlice(frame)
        ip = ts_imp.getProcessor()
        stats = ip.getStatistics()
        total_intensity = stats.mean * stats.area  # Cumulative intensity
        
        if total_intensity < min_intensity:
            min_intensity = total_intensity
            min_frame = frame
        if total_intensity > max_intensity:
            max_intensity = total_intensity
            max_frame = frame
    
    IJ.showProgress(1.0)
    IJ.log("  Diastole (min): frame %d" % min_frame)
    IJ.log("  Systole (max): frame %d" % max_frame)
    
    # Get the min and max calcium frames
    ts_imp.setSlice(min_frame)
    calcium_min_ip = ts_imp.getProcessor().duplicate()
    ts_imp.setSlice(max_frame)
    calcium_max_ip = ts_imp.getProcessor().duplicate()
    
    # === Create 4-channel x 2-timepoint hyperstack ===
    # Stack order for hyperstack: channels vary fastest, then slices, then frames
    # We want: 4 channels x 1 slice x 2 frames
    IJ.log("Creating 4-channel x 2-timepoint hyperstack...")
    
    width, height = binned_static.getWidth(), binned_static.getHeight()
    merged_stack = ImageStack(width, height)
    static_stack = binned_static.getStack()
    
    # Timepoint 1: Diastole (min calcium)
    for c in range(1, 4):  # Static channels
        merged_stack.addSlice("C%d_T1" % c, static_stack.getProcessor(c).duplicate())
    merged_stack.addSlice("Calcium_T1", calcium_min_ip)
    
    # Timepoint 2: Systole (max calcium)
    for c in range(1, 4):  # Static channels (duplicated)
        merged_stack.addSlice("C%d_T2" % c, static_stack.getProcessor(c).duplicate())
    merged_stack.addSlice("Calcium_T2", calcium_max_ip)
    
    merged_imp = ImagePlus("4Ch_2T_Merged", merged_stack)
    # setDimensions(nChannels, nSlices, nFrames)
    merged_imp.setDimensions(4, 1, 2)
    merged_composite = CompositeImage(merged_imp, CompositeImage.COMPOSITE)
    
    binned_static.close()
    IJ.log("4-channel x 2-timepoint hyperstack created (T1=Diastole, T2=Systole)")
    return merged_composite, ts_imp


def step_setup_composite_view(merged_imp):
    """Step 3: Setup composite view with LUTs."""
    apply_composite_luts(merged_imp, n_channels=4)
    auto_contrast(merged_imp)
    merged_imp.show()
    merged_imp.setTitle("4Ch_Diastole_Systole")
    
    IJ.run("Brightness/Contrast...")
    IJ.run("Channels Tool...")
    
    WaitForUserDialog("Step 3: Adjust Visualization",
        "Adjust brightness/contrast as needed.\n\n"
        "Channels:\n"
        "  Grays (Ch1): Actin\n"
        "  Cyan (Ch2): G0/G1 nuclei\n"
        "  Magenta (Ch3): S/G2/M nuclei\n"
        "  Green (Ch4): Calcium\n\n"
        "TIMEPOINTS (use slider at bottom):\n"
        "  T1 = Diastole (low calcium)\n"
        "  T2 = Systole (high calcium)\n\n"
        "Toggle between T1/T2 to see cells with/without calcium!\n\n"
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
    dff_traces = calculate_dff(traces, bg_names)
    
    IJ.log("Plotting...")
    plot_traces(dff_traces, "Calcium Transients - dF/F0")
    
    IJ.log("Exporting traces (raw + normalized)...")
    export_traces_csv(traces, dff_traces, output_dir)
    generate_summary(dff_traces, os.path.join(output_dir, "transient_summary.csv"))
    
    return True


def step_finalize(output_dir):
    """Step 8: Save and finish."""
    rm = get_roi_manager()
    save_rois(rm, os.path.join(output_dir, "calcium_rois.zip"))
    
    IJ.showMessage("Analysis Complete",
        "Files saved to:\n" + output_dir + "\n\n"
        "- calcium_rois.zip\n"
        "- calcium_traces_raw.csv (raw intensities)\n"
        "- calcium_traces_dff.csv (dF/F0 normalized + stats)\n"
        "- transient_summary.csv")
    IJ.log("Done!")


# =============================================================================
# MAIN WIZARD
# =============================================================================

def run_wizard():
    """Main entry point - V1 AUTO with automated ROI selection."""
    IJ.log("=" * 50)
    IJ.log("Calcium Analysis Wizard v1-AUTO")
    IJ.log("Automated nuclear segmentation")
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
    
    # Show the composite for reference
    composite_imp = step_setup_composite_view(merged_imp)
    
    # === AUTOMATED ROI SELECTION ===
    IJ.log("")
    IJ.log("Starting automated ROI detection...")
    
    # Run auto-segmentation
    cyan_rois, magenta_rois, bg_rois, avg_area = auto_segment_all_rois(composite_imp, ts_imp)
    
    # Add ROIs to manager
    rm = get_roi_manager()
    rm.reset()  # Clear any existing ROIs
    
    add_rois_to_manager(cyan_rois, "Cyan", rm)
    add_rois_to_manager(magenta_rois, "Magenta", rm)
    add_rois_to_manager(bg_rois, "BG", rm)
    
    IJ.log("")
    IJ.log("ROI Summary:")
    IJ.log("  Cyan (G0/G1): %d" % len(cyan_rois))
    IJ.log("  Magenta (S/G2/M): %d" % len(magenta_rois))
    IJ.log("  Background: %d" % len(bg_rois))
    
    # Save verification image
    IJ.log("")
    IJ.log("Saving ROI verification image...")
    verification_imp = save_roi_verification_image(composite_imp, rm, output_dir)
    
    # Show ROI Manager
    rm.setVisible(True)
    
    # Ask user to verify
    gd = GenericDialog("Automated ROI Detection Complete")
    gd.addMessage("ROIs detected automatically:")
    gd.addMessage("  - Cyan (G0/G1): %d nuclei" % len(cyan_rois))
    gd.addMessage("  - Magenta (S/G2/M): %d nuclei" % len(magenta_rois))
    gd.addMessage("  - Background: %d regions" % len(bg_rois))
    gd.addMessage("")
    gd.addMessage("Review the ROI verification image and ROI Manager.")
    gd.addMessage("Click OK to proceed with trace extraction,")
    gd.addMessage("or Cancel to abort.")
    gd.showDialog()
    
    if gd.wasCanceled():
        IJ.log("User cancelled after ROI review.")
        return
    
    # Extract and analyze
    if not step_extract_and_analyze(ts_imp, output_dir):
        return
    
    # Close verification image
    verification_imp.close()
    
    step_finalize(output_dir)


# Run wizard
if __name__ == "__main__" or __name__ == "__builtin__":
    run_wizard()
