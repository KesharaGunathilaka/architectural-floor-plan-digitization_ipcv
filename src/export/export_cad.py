"""
Vectorization & Export Script — Stage 5

PURPOSE:
    Converts post-processed YOLO detections into structured JSON and
    AutoCAD-compatible DXF files.  Integrates directly with the
    post_process.py pipeline so that a single command runs inference
    on a floor plan image and exports a fully viewable CAD file.

FIXES (from blank-DXF issue):
    1. The old __main__ used hardcoded sample data → now runs real inference.
    2. $EXTMIN/$EXTMAX were unset → now explicitly computed and written
       so that CAD viewers auto-zoom to the drawing extents.
    3. Added text labels for each element in the DXF.
    4. Added all 6 class layers (was missing Staircase/Toilet/Sink layers).
    5. Added INSUNITS=6 (meters) in the header so CAD reads correct scale.

HOW TO RUN (standalone full pipeline):
    python src/export/export_cad.py                     # Processes default test image
    python src/export/export_cad.py <image_path>        # Processes any image
"""

import json
import sys
from pathlib import Path

import cv2
import ezdxf

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.postprocessing.post_process import process_floorplan

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_WEIGHTS = "models/final/best.pt"
OUTPUT_DIR = Path("runs/exports")
DEFAULT_TEST_IMAGE = "data/yolo_dataset_processed/images/test/high_quality_architectural_10499_F1.png"

# Layer configuration: class_name → (layer_name, DXF color index)
LAYER_CONFIG = {
    "Door":      ("DOORS",      3),   # Green
    "Window":    ("WINDOWS",    5),   # Blue
    "Wall":      ("WALLS",      1),   # Red
    "Staircase": ("STAIRCASES", 4),   # Cyan
    "Toilet":    ("TOILETS",    6),   # Magenta
    "Sink":      ("SINKS",      2),   # Yellow
}


def export_to_json(elements: list, output_path: str):
    """Saves the extracted architectural elements to a structured JSON file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        json.dump({"architectural_elements": elements}, f, indent=4)

    print(f"  JSON export: {out_file}")


def export_to_dxf(elements: list, output_path: str, img_height_px: float, ppm: float):
    """
    Generates a DXF CAD file from detected architectural elements.

    Key improvements:
      - Sets $EXTMIN/$EXTMAX so CAD viewers auto-zoom to content.
      - Sets $INSUNITS=6 (meters) for correct scaling.
      - Adds a grey WALLS_CENTERLINE layer for snapped wall axes.
      - Renders high-quality custom symbols for doors, windows, staircases, toilets, and sinks.
      - Places text labels on a hidden-by-default layer (DETECTION_LABELS) to avoid clutter.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a new DXF document (R2010 = AutoCAD 2010 format, widely supported)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    # Setup CAD Layers — one per class with distinct colors
    for class_name, (layer_name, color) in LAYER_CONFIG.items():
        doc.layers.add(layer_name, color=color)
    doc.layers.add("WALLS_CENTERLINE", color=8) # Grey for wall axes
    labels_layer = doc.layers.add("DETECTION_LABELS", color=7)  # White — for text annotations
    labels_layer.off() # Disabled by default for clean presentation

    # Track drawing extents for viewport fix
    all_x, all_y = [], []

    for det in elements:
        class_name = det["class_name"]
        layer_name, _ = LAYER_CONFIG.get(class_name, ("FIXTURES", 6))

        # Invert Y-axis (image top-left → CAD bottom-left origin)
        cad_y1_px = img_height_px - det["y2"]
        cad_y2_px = img_height_px - det["y1"]

        # Convert pixel coordinates to real-world meters
        x1_m = det["x1"] / ppm
        y1_m = cad_y1_px / ppm
        x2_m = det["x2"] / ppm
        y2_m = cad_y2_px / ppm

        # Track extents
        all_x.extend([x1_m, x2_m])
        all_y.extend([y1_m, y2_m])

        # ── Draw Class-Specific Architectural Symbols ─────────────────────
        if class_name == "Wall":
            # 1. Boundary box (double line boundary)
            points = [
                (x1_m, y1_m),
                (x2_m, y1_m),
                (x2_m, y2_m),
                (x1_m, y2_m),
                (x1_m, y1_m),
            ]
            msp.add_lwpolyline(points, dxfattribs={"layer": layer_name})

            # 2. Snapped Centerline (if available)
            if "centerline" in det:
                c = det["centerline"]
                c_y1 = img_height_px - c["y2"] # note: Y inversion for points
                c_y2 = img_height_px - c["y1"]
                msp.add_line(
                    (c["x1"] / ppm, c_y1 / ppm),
                    (c["x2"] / ppm, c_y2 / ppm),
                    dxfattribs={"layer": "WALLS_CENTERLINE"}
                )

        elif class_name == "Door":
            w = x2_m - x1_m
            h = y2_m - y1_m
            y_mid = (y1_m + y2_m) / 2
            x_mid = (x1_m + x2_m) / 2

            if w >= h: # Horizontal door opening
                # Hinge at left (x1_m), swings up
                msp.add_line((x1_m, y_mid), (x1_m, y_mid + w), dxfattribs={"layer": layer_name}) # door leaf
                msp.add_arc(center=(x1_m, y_mid), radius=w, start_angle=0, end_angle=90, dxfattribs={"layer": layer_name}) # swing arc
                # Reference threshold line
                msp.add_line((x1_m, y_mid), (x2_m, y_mid), dxfattribs={"layer": layer_name})
            else: # Vertical door opening
                # Hinge at bottom (y1_m), swings right
                msp.add_line((x_mid, y1_m), (x_mid + h, y1_m), dxfattribs={"layer": layer_name}) # door leaf
                msp.add_arc(center=(x_mid, y1_m), radius=h, start_angle=0, end_angle=90, dxfattribs={"layer": layer_name}) # swing arc
                # Reference threshold line
                msp.add_line((x_mid, y1_m), (x_mid, y2_m), dxfattribs={"layer": layer_name})

        elif class_name == "Window":
            # Outer boundary box
            points = [(x1_m, y1_m), (x2_m, y1_m), (x2_m, y2_m), (x1_m, y2_m), (x1_m, y1_m)]
            msp.add_lwpolyline(points, dxfattribs={"layer": layer_name})
            
            # Central sill line
            w = x2_m - x1_m
            h = y2_m - y1_m
            if w >= h:
                y_mid = (y1_m + y2_m) / 2
                msp.add_line((x1_m, y_mid), (x2_m, y_mid), dxfattribs={"layer": layer_name})
            else:
                x_mid = (x1_m + x2_m) / 2
                msp.add_line((x_mid, y1_m), (x_mid, y2_m), dxfattribs={"layer": layer_name})

        elif class_name == "Toilet":
            w = x2_m - x1_m
            h = y2_m - y1_m
            if h >= w: # Portrait orientation
                # Tank at the top (y2_m)
                tank_height = min(h * 0.3, 0.2)
                tank_pts = [
                    (x1_m, y2_m - tank_height),
                    (x2_m, y2_m - tank_height),
                    (x2_m, y2_m),
                    (x1_m, y2_m),
                    (x1_m, y2_m - tank_height)
                ]
                msp.add_lwpolyline(tank_pts, dxfattribs={"layer": layer_name})
                # Bowl circle
                bowl_radius = min(w * 0.4, (h - tank_height) * 0.5)
                bowl_center = ((x1_m + x2_m) / 2, y2_m - tank_height - bowl_radius)
                msp.add_circle(bowl_center, bowl_radius, dxfattribs={"layer": layer_name})
            else: # Landscape orientation
                # Tank at the left (x1_m)
                tank_width = min(w * 0.3, 0.2)
                tank_pts = [
                    (x1_m, y1_m),
                    (x1_m + tank_width, y1_m),
                    (x1_m + tank_width, y2_m),
                    (x1_m, y2_m),
                    (x1_m, y1_m)
                ]
                msp.add_lwpolyline(tank_pts, dxfattribs={"layer": layer_name})
                # Bowl circle
                bowl_radius = min(h * 0.4, (w - tank_width) * 0.5)
                bowl_center = (x1_m + tank_width + bowl_radius, (y1_m + y2_m) / 2)
                msp.add_circle(bowl_center, bowl_radius, dxfattribs={"layer": layer_name})

        elif class_name == "Sink":
            # Outer basin boundary
            outer_pts = [(x1_m, y1_m), (x2_m, y1_m), (x2_m, y2_m), (x1_m, y2_m), (x1_m, y1_m)]
            msp.add_lwpolyline(outer_pts, dxfattribs={"layer": layer_name})
            
            # Inner bowl boundary (offset by 10% or max 5cm)
            w, h = x2_m - x1_m, y2_m - y1_m
            offset_x = min(w * 0.1, 0.05)
            offset_y = min(h * 0.1, 0.05)
            ix1, ix2 = x1_m + offset_x, x2_m - offset_x
            iy1, iy2 = y1_m + offset_y, y2_m - offset_y
            inner_pts = [(ix1, iy1), (ix2, iy1), (ix2, iy2), (ix1, iy2), (ix1, iy1)]
            msp.add_lwpolyline(inner_pts, dxfattribs={"layer": layer_name})
            
            # Drain circle
            center_x = (x1_m + x2_m) / 2
            center_y = (y1_m + y2_m) / 2
            drain_radius = min(w, h) * 0.08
            msp.add_circle((center_x, center_y), drain_radius, dxfattribs={"layer": layer_name})
            
            # Faucet indicator line
            if h >= w:
                msp.add_line((center_x, iy2), (center_x, iy2 - drain_radius * 1.5), dxfattribs={"layer": layer_name})
            else:
                msp.add_line((ix1, center_y), (ix1 + drain_radius * 1.5, center_y), dxfattribs={"layer": layer_name})

        elif class_name == "Staircase":
            # Outer box boundary
            outer_pts = [(x1_m, y1_m), (x2_m, y1_m), (x2_m, y2_m), (x1_m, y2_m), (x1_m, y1_m)]
            msp.add_lwpolyline(outer_pts, dxfattribs={"layer": layer_name})
            
            w, h = x2_m - x1_m, y2_m - y1_m
            step_tread = 0.25 # standard step size
            
            if h >= w: # Vertical staircase
                num_steps = int(h / step_tread)
                if num_steps < 2:
                    num_steps = 5
                actual_step_h = h / num_steps
                for i in range(1, num_steps):
                    curr_y = y1_m + i * actual_step_h
                    msp.add_line((x1_m, curr_y), (x2_m, curr_y), dxfattribs={"layer": layer_name})
                # Direction arrow
                cx = (x1_m + x2_m) / 2
                msp.add_line((cx, y1_m + actual_step_h), (cx, y2_m - actual_step_h), dxfattribs={"layer": layer_name})
                arrow_size = min(w * 0.15, actual_step_h * 0.5)
                msp.add_line((cx, y2_m - actual_step_h), (cx - arrow_size, y2_m - actual_step_h - arrow_size), dxfattribs={"layer": layer_name})
                msp.add_line((cx, y2_m - actual_step_h), (cx + arrow_size, y2_m - actual_step_h - arrow_size), dxfattribs={"layer": layer_name})
            else: # Horizontal staircase
                num_steps = int(w / step_tread)
                if num_steps < 2:
                    num_steps = 5
                actual_step_w = w / num_steps
                for i in range(1, num_steps):
                    curr_x = x1_m + i * actual_step_w
                    msp.add_line((curr_x, y1_m), (curr_x, y2_m), dxfattribs={"layer": layer_name})
                # Direction arrow
                cy = (y1_m + y2_m) / 2
                msp.add_line((x1_m + actual_step_w, cy), (x2_m - actual_step_w, cy), dxfattribs={"layer": layer_name})
                arrow_size = min(h * 0.15, actual_step_w * 0.5)
                msp.add_line((x2_m - actual_step_w, cy), (x2_m - actual_step_w - arrow_size, cy - arrow_size), dxfattribs={"layer": layer_name})
                msp.add_line((x2_m - actual_step_w, cy), (x2_m - actual_step_w - arrow_size, cy + arrow_size), dxfattribs={"layer": layer_name})

        else: # Fallback to standard polyline
            points = [(x1_m, y1_m), (x2_m, y1_m), (x2_m, y2_m), (x1_m, y2_m), (x1_m, y1_m)]
            msp.add_lwpolyline(points, dxfattribs={"layer": layer_name})

        # ── Toggleable Confidence Metadata labels (invisible by default) ──
        label = f"{class_name} ({det['confidence']:.0%})"
        text_height = max(0.06, min((y2_m - y1_m) * 0.25, 0.2)) # clean, smaller text
        msp.add_text(
            label,
            dxfattribs={
                "layer": "DETECTION_LABELS",
                "height": text_height,
                "insert": (x1_m, y2_m + 0.03),
            },
        )

    # ── Fix viewport extents ──────────────────────────────────────────────
    # This is the KEY fix: without valid $EXTMIN/$EXTMAX, CAD viewers
    # don't know where the drawing content is, so they show a blank canvas.
    if all_x and all_y:
        margin = 0.5  # Add 0.5m margin around the drawing
        min_x, max_x = min(all_x) - margin, max(all_x) + margin
        min_y, max_y = min(all_y) - margin, max(all_y) + margin

        doc.header["$EXTMIN"] = (min_x, min_y, 0)
        doc.header["$EXTMAX"] = (max_x, max_y, 0)
        doc.header["$LIMMIN"] = (min_x, min_y)
        doc.header["$LIMMAX"] = (max_x, max_y)

    # Set drawing units to meters
    doc.header["$INSUNITS"] = 6  # 6 = meters in DXF standard

    # Save the document
    doc.saveas(str(out_file))
    print(f"  DXF export:  {out_file}")
    if all_x:
        print(f"  Drawing extents: ({min(all_x):.1f}, {min(all_y):.1f}) to ({max(all_x):.1f}, {max(all_y):.1f}) meters")
        print(f"  Total elements: {len(elements)}")


def run_full_pipeline(image_path: str, model_path: str = MODEL_WEIGHTS):
    """
    End-to-end pipeline: Image → YOLO Inference → Post-Processing → CAD Export

    This is the function that was MISSING — the old code had post_process.py
    and export_cad.py as disconnected scripts.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    print(f"\n{'='*60}")
    print("  FLOOR PLAN → CAD PIPELINE")
    print(f"  Image: {image_path.name}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    # Step 1: Run inference + post-processing
    print("\n[1/3] Running YOLO inference + post-processing...")
    elements, ppm = process_floorplan(str(image_path), model_path)

    if not elements:
        print("WARNING: No architectural elements detected! CAD file will be empty.")
        print("Check that the model path is correct and the image is a valid floor plan.")
        return

    # Step 2: Get image dimensions for Y-axis inversion
    img = cv2.imread(str(image_path))
    img_height_px = img.shape[0]

    # Step 3: Export
    stem = image_path.stem
    output_dir = OUTPUT_DIR / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2/3] Exporting to JSON...")
    export_to_json(elements, str(output_dir / f"{stem}.json"))

    print("[3/3] Exporting to DXF...")
    export_to_dxf(elements, str(output_dir / f"{stem}.dxf"), img_height_px, ppm)

    print(f"\n{'='*60}")
    print("  EXPORT COMPLETE")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")

    return elements


if __name__ == "__main__":
    # Accept optional image path from command line
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    else:
        input_image = DEFAULT_TEST_IMAGE

    run_full_pipeline(input_image)
