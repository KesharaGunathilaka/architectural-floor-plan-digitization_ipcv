"""
Post-Processing & Spatial Mapping Script — Stage 4

PURPOSE:
    Takes raw YOLO predictions, applies NMS/Confidence filters,
    runs geometric sanity checks, and calculates real-world scale (PPM).

HOW TO RUN:
    python src/postprocessing/post_process.py
"""

from pathlib import Path
from ultralytics import YOLO

# Architectural Constants
STANDARD_DOOR_WIDTH_M = 0.9  # Standard residential door is ~900mm
DOOR_CLASS_ID = 0  # Ensure this matches your dataset.yaml
WALL_CLASS_ID = 2


def calculate_ppm(detections) -> float:
    """
    Calculates Pixels-Per-Meter (PPM) based on detected doors.
    Accounts for both horizontal and vertical door orientations.
    """
    door_lengths = []
    for det in detections:
        if det["class_id"] == DOOR_CLASS_ID:
            # The 0.9m opening will always be the longest dimension of the bounding box
            actual_opening_px = max(det["width_px"], det["height_px"])
            door_lengths.append(actual_opening_px)

    if not door_lengths:
        print("[WARNING] No doors detected. Spatial mapping will be uncalibrated.")
        return 100.0  # Safe fallback

    # Average the length of all detected doors to smooth out pixel jitter
    avg_door_px = sum(door_lengths) / len(door_lengths)
    ppm = avg_door_px / STANDARD_DOOR_WIDTH_M
    return ppm


def geometric_sanity_check(det) -> bool:
    """
    Filters out mathematically impossible architectural elements.
    Returns True if the detection makes sense, False if it should be discarded.
    """
    w, h = det["width_px"], det["height_px"]
    aspect_ratio = max(w, h) / min(w, h)

    # Example check: Walls shouldn't be perfect squares
    if det["class_id"] == WALL_CLASS_ID and aspect_ratio < 1.5:
        return False

    return True


def refine_geometry(detections, ppm):
    """
    Geometric refinement of floor plan elements:
      - Classifies walls into horizontal and vertical.
      - Merges collinear/overlapping walls of the same class.
      - Snaps wall endpoints at corners and T-junctions (centerline-based).
      - Aligns doors/windows to be centered within their host walls.
    """
    walls = [d for d in detections if d["class_name"] == "Wall"]
    doors = [d for d in detections if d["class_name"] == "Door"]
    windows = [d for d in detections if d["class_name"] == "Window"]
    others = [d for d in detections if d["class_name"] not in ["Wall", "Door", "Window"]]

    # 1. Classify walls as Horizontal or Vertical
    h_walls = []
    v_walls = []
    for w in walls:
        width = w["x2"] - w["x1"]
        height = w["y2"] - w["y1"]
        if width >= height:
            h_walls.append(w)
        else:
            v_walls.append(w)

    tol_collinear = 0.25 * ppm
    tol_gap = 0.25 * ppm

    # 2. Merge collinear/overlapping walls
    def merge_wall_list(wall_list, orientation):
        merged = True
        current = list(wall_list)
        while merged:
            merged = False
            next_list = []
            used = set()
            for i in range(len(current)):
                if i in used:
                    continue
                w1 = current[i]
                merged_any = False
                for j in range(i + 1, len(current)):
                    if j in used:
                        continue
                    w2 = current[j]

                    if orientation == "horizontal":
                        y1_mid = (w1["y1"] + w1["y2"]) / 2
                        y2_mid = (w2["y1"] + w2["y2"]) / 2
                        is_collinear = abs(y1_mid - y2_mid) <= tol_collinear

                        x1_min = min(w1["x1"], w2["x1"])
                        x2_max = max(w1["x2"], w2["x2"])
                        len1 = w1["x2"] - w1["x1"]
                        len2 = w2["x2"] - w2["x1"]
                        union_len = x2_max - x1_min
                        sum_len = len1 + len2

                        is_close = union_len <= sum_len + tol_gap

                        if is_collinear and is_close:
                            new_x1 = x1_min
                            new_x2 = x2_max
                            y_mid = (y1_mid * len1 + y2_mid * len2) / (len1 + len2)
                            h = max(w1["y2"] - w1["y1"], w2["y2"] - w2["y1"])
                            
                            w_merged = {
                                "class_id": w1["class_id"],
                                "class_name": w1["class_name"],
                                "confidence": max(w1["confidence"], w2["confidence"]),
                                "x1": new_x1,
                                "y1": y_mid - h/2,
                                "x2": new_x2,
                                "y2": y_mid + h/2,
                                "width_px": new_x2 - new_x1,
                                "height_px": h,
                            }
                            current[j] = w_merged
                            used.add(i)
                            merged = True
                            merged_any = True
                            break
                    else:
                        x1_mid = (w1["x1"] + w1["x2"]) / 2
                        x2_mid = (w2["x1"] + w2["x2"]) / 2
                        is_collinear = abs(x1_mid - x2_mid) <= tol_collinear

                        y1_min = min(w1["y1"], w2["y1"])
                        y2_max = max(w1["y2"], w2["y2"])
                        len1 = w1["y2"] - w1["y1"]
                        len2 = w2["y2"] - w2["y1"]
                        union_len = y2_max - y1_min
                        sum_len = len1 + len2

                        is_close = union_len <= sum_len + tol_gap

                        if is_collinear and is_close:
                            new_y1 = y1_min
                            new_y2 = y2_max
                            x_mid = (x1_mid * len1 + x2_mid * len2) / (len1 + len2)
                            w_thickness = max(w1["x2"] - w1["x1"], w2["x2"] - w2["x1"])

                            w_merged = {
                                "class_id": w1["class_id"],
                                "class_name": w1["class_name"],
                                "confidence": max(w1["confidence"], w2["confidence"]),
                                "x1": x_mid - w_thickness/2,
                                "y1": new_y1,
                                "x2": x_mid + w_thickness/2,
                                "y2": new_y2,
                                "width_px": w_thickness,
                                "height_px": new_y2 - new_y1,
                            }
                            current[j] = w_merged
                            used.add(i)
                            merged = True
                            merged_any = True
                            break
                if not merged_any:
                    next_list.append(w1)
            current = next_list
        return current

    merged_h = merge_wall_list(h_walls, "horizontal")
    merged_v = merge_wall_list(v_walls, "vertical")

    # 3. Snap centerlines of H and V walls
    h_lines = []
    for w in merged_h:
        h_lines.append({
            "y": (w["y1"] + w["y2"]) / 2,
            "x1": w["x1"],
            "x2": w["x2"],
            "t": w["y2"] - w["y1"],
            "conf": w["confidence"],
            "class_id": w["class_id"]
        })
    v_lines = []
    for w in merged_v:
        v_lines.append({
            "x": (w["x1"] + w["x2"]) / 2,
            "y1": w["y1"],
            "y2": w["y2"],
            "t": w["x2"] - w["x1"],
            "conf": w["confidence"],
            "class_id": w["class_id"]
        })

    snap_tol = 0.25 * ppm

    # Snap H endpoints to V centerlines
    for h in h_lines:
        hy = h["y"]
        best_v_left = None
        min_dist_left = snap_tol
        best_v_right = None
        min_dist_right = snap_tol

        for v in v_lines:
            vx = v["x"]
            if v["y1"] - snap_tol <= hy <= v["y2"] + snap_tol:
                d_x1 = abs(h["x1"] - vx)
                if d_x1 < min_dist_left:
                    min_dist_left = d_x1
                    best_v_left = v

                d_x2 = abs(h["x2"] - vx)
                if d_x2 < min_dist_right:
                    min_dist_right = d_x2
                    best_v_right = v

        if best_v_left:
            h["x1"] = best_v_left["x"]
            if hy < best_v_left["y1"]:
                best_v_left["y1"] = hy
            elif hy > best_v_left["y2"]:
                best_v_left["y2"] = hy

        if best_v_right:
            h["x2"] = best_v_right["x"]
            if hy < best_v_right["y1"]:
                best_v_right["y1"] = hy
            elif hy > best_v_right["y2"]:
                best_v_right["y2"] = hy

    # Snap V endpoints to H centerlines
    for v in v_lines:
        vx = v["x"]
        best_h_top = None
        min_dist_top = snap_tol
        best_h_bottom = None
        min_dist_bottom = snap_tol

        for h in h_lines:
            hy = h["y"]
            if h["x1"] - snap_tol <= vx <= h["x2"] + snap_tol:
                d_y1 = abs(v["y1"] - hy)
                if d_y1 < min_dist_top:
                    min_dist_top = d_y1
                    best_h_top = h

                d_y2 = abs(v["y2"] - hy)
                if d_y2 < min_dist_bottom:
                    min_dist_bottom = d_y2
                    best_h_bottom = h

        if best_h_top:
            v["y1"] = best_h_top["y"]
            if vx < best_h_top["x1"]:
                best_h_top["x1"] = vx
            elif vx > best_h_top["x2"]:
                best_h_top["x2"] = vx

        if best_h_bottom:
            v["y2"] = best_h_bottom["y"]
            if vx < best_h_bottom["x1"]:
                best_h_bottom["x1"] = vx
            elif vx > best_h_bottom["x2"]:
                best_h_bottom["x2"] = vx

    # Reconstruct walls from snapped lines
    refined_walls = []
    for h in h_lines:
        refined_walls.append({
            "class_id": h["class_id"],
            "class_name": "Wall",
            "confidence": h["conf"],
            "x1": h["x1"],
            "y1": h["y"] - h["t"]/2,
            "x2": h["x2"],
            "y2": h["y"] + h["t"]/2,
            "width_px": h["x2"] - h["x1"],
            "height_px": h["t"],
            "centerline": {
                "x1": h["x1"],
                "y1": h["y"],
                "x2": h["x2"],
                "y2": h["y"]
            }
        })
    for v in v_lines:
        refined_walls.append({
            "class_id": v["class_id"],
            "class_name": "Wall",
            "confidence": v["conf"],
            "x1": v["x"] - v["t"]/2,
            "y1": v["y1"],
            "x2": v["x"] + v["t"]/2,
            "y2": v["y2"],
            "width_px": v["t"],
            "height_px": v["y2"] - v["y1"],
            "centerline": {
                "x1": v["x"],
                "y1": v["y1"],
                "x2": v["x"],
                "y2": v["y2"]
            }
        })

    # 4. Align doors and windows to walls
    align_tol = 0.25 * ppm
    aligned_doors_wins = []
    for item in (doors + windows):
        w_px = item["x2"] - item["x1"]
        h_px = item["y2"] - item["y1"]
        is_horiz = w_px >= h_px

        item_x_mid = (item["x1"] + item["x2"]) / 2
        item_y_mid = (item["y1"] + item["y2"]) / 2

        best_wall = None
        min_dist = align_tol

        for wall in refined_walls:
            wall_w = wall["x2"] - wall["x1"]
            wall_h = wall["y2"] - wall["y1"]
            wall_is_horiz = wall_w >= wall_h
            if wall_is_horiz != is_horiz:
                continue

            if is_horiz:
                wall_y_mid = (wall["y1"] + wall["y2"]) / 2
                if wall["x1"] - align_tol <= item_x_mid <= wall["x2"] + align_tol:
                    dist = abs(item_y_mid - wall_y_mid)
                    if dist < min_dist:
                        min_dist = dist
                        best_wall = wall
            else:
                wall_x_mid = (wall["x1"] + wall["x2"]) / 2
                if wall["y1"] - align_tol <= item_y_mid <= wall["y2"] + align_tol:
                    dist = abs(item_x_mid - wall_x_mid)
                    if dist < min_dist:
                        min_dist = dist
                        best_wall = wall

        new_item = dict(item)
        if best_wall:
            if is_horiz:
                wall_y_mid = (best_wall["y1"] + best_wall["y2"]) / 2
                thickness = best_wall["y2"] - best_wall["y1"]
                new_item["y1"] = wall_y_mid - thickness / 2
                new_item["y2"] = wall_y_mid + thickness / 2
                new_item["height_px"] = thickness
            else:
                wall_x_mid = (best_wall["x1"] + best_wall["x2"]) / 2
                thickness = best_wall["x2"] - best_wall["x1"]
                new_item["x1"] = wall_x_mid - thickness / 2
                new_item["x2"] = wall_x_mid + thickness / 2
                new_item["width_px"] = thickness
        aligned_doors_wins.append(new_item)

    return refined_walls + aligned_doors_wins + others


def process_floorplan(
    image_path: str, model_path: str, conf_thresh=0.25, iou_thresh=0.45
):
    """
    Runs the full Stage 4 pipeline on a single image.
    """
    print(f"Processing: {Path(image_path).name}")

    # 1. Load Model & Run Inference (NMS and Conf filtering happen here)
    model = YOLO(model_path)
    results = model.predict(
        source=image_path, conf=conf_thresh, iou=iou_thresh, verbose=False
    )[0]

    # Extract raw bounding boxes
    boxes = results.boxes
    names = model.names

    raw_detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        conf = box.conf[0].item()

        raw_detections.append(
            {
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width_px": x2 - x1,
                "height_px": y2 - y1,
            }
        )

    # 2. Class-Level Validation (Geometric Sanity Checks)
    filtered_detections = [d for d in raw_detections if geometric_sanity_check(d)]

    # 3. Spatial Mapping (Calculate PPM)
    ppm = calculate_ppm(filtered_detections)
    print(f"Calculated Scale: {ppm:.2f} Pixels Per Meter")

    # 4. Geometric Refinement (Wall Merging, Snapping, and Alignment)
    refined_detections = refine_geometry(filtered_detections, ppm)

    # 5. Attach Real-World Coordinates
    final_elements = []
    for det in refined_detections:
        det["width_m"] = det["width_px"] / ppm
        det["height_m"] = det["height_px"] / ppm
        final_elements.append(det)

    print(
        f"Retained {len(final_elements)} valid architectural elements after refinement."
    )
    return final_elements, ppm


if __name__ == "__main__":
    # Test the pipeline on a single image
    TEST_IMAGE = "data/yolo_dataset_processed/images/test/colorful_10711_F1.png"
    MODEL_WEIGHTS = "models/final/best.pt"

    if Path(TEST_IMAGE).exists():
        elements, scale = process_floorplan(TEST_IMAGE, MODEL_WEIGHTS)

        # Print a sample of the cleaned data
        if elements:
            print("\nSample Extracted Element:")
            print(elements[0])
    else:
        print("Please update the TEST_IMAGE path to a valid test image.")
