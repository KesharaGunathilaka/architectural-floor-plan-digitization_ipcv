# Project Overview

**Automated Digitization of Architectural Floor Plans**

A computer-vision pipeline that converts static floor-plan images (scanned
blueprints, PDFs) into structured, editable CAD data — detecting architectural
elements with YOLOv11 and exporting them to JSON and AutoCAD-compatible DXF.

---

## 1. The Problem

Millions of architectural floor plans exist only as raster images. To a
computer these are just pixels — there is no machine-readable record of where
the walls, doors, or fixtures are. Re-drawing a plan in CAD by hand is slow,
expensive, and error-prone.

This project automates that conversion: **image in → CAD file out.**

## 2. What It Detects

Six architectural element classes:

| ID | Class     | Notes                                            |
|----|-----------|--------------------------------------------------|
| 0  | Door      | Used to calibrate real-world scale (≈0.9 m wide) |
| 1  | Window    |                                                  |
| 2  | Wall      | Most frequent class (~54% of annotations)        |
| 3  | Staircase | Encoded as `Stairs` in the source SVGs           |
| 4  | Toilet    | Rare class (~3%)                                 |
| 5  | Sink      | Rare class (~3%)                                 |

## 3. The Dataset

[**CubiCasa5k**](https://github.com/CubiCasa/CubiCasa5k) — 5,000 annotated
residential floor plans across three quality categories (`colorful`,
`high_quality`, `high_quality_architectural`). Annotations are stored as SVG
vector shapes, which the pipeline converts to YOLO bounding boxes.

The dataset is **not** included in this repository. Download it separately and
point `CUBICASA_ROOT` at it in your `.env`.

**Class imbalance** is significant (Wall:Staircase ≈ 28:1), mitigated during
training via class-loss weighting, mosaic augmentation, and copy-paste
oversampling of rare classes.

## 4. Architecture / Pipeline

The system is a seven-phase pipeline. Each phase is a standalone script
invoked through `make`.

```
┌─────────────┐   Phase 1            ┌──────────────┐   Phase 2
│ Raw Floor   │  SVG audit →         │ YOLO dataset │  SVG → YOLO
│ Plans (SVG  │  dataset stats →     │ (images +    │  bbox conversion
│ + PNG)      │  stratified split    │ .txt labels) │  + verification
└─────────────┘                      └──────────────┘
        │                                    │
        ▼ Phase 3                            ▼ Phase 5
┌──────────────┐  Letterbox to       ┌──────────────┐  Train YOLOv11l
│ Preprocessed │  1280×1280,         │ Trained      │  (mosaic, copy-paste,
│ images       │  recompute boxes    │ model        │  class weighting)
└──────────────┘                     └──────────────┘
        │                                    │
        ▼ Phase 6                            ▼ Phase 7
┌──────────────┐  Test-set mAP,      ┌──────────────┐  Post-process →
│ Evaluation   │  per-class AP,      │ JSON + DXF   │  geometry refine →
│ report       │  visual inference   │ CAD export   │  vectorize
└──────────────┘                     └──────────────┘
```

### Phase breakdown

| Phase | Stage              | Key script                              |
|-------|--------------------|-----------------------------------------|
| 1     | SVG audit          | `src/data/audit_svg.py`                 |
| 1     | Dataset statistics | `src/data/dataset_stats.py`             |
| 1     | Subset selection   | `src/data/select_subset.py`             |
| 2     | Annotation convert | `src/data/convert_annotations.py`       |
| 2     | Verify annotations | `src/data/verify_annotations.py`        |
| 3     | Preprocessing      | `src/preprocessing/preprocess_pipeline.py` |
| 5     | Training           | `src/training/train_yolo.py`            |
| 5     | Hyperparam tuning  | `src/training/hyperparameter_tuning.py` |
| 6     | Evaluation         | `src/evaluation/evaluate_test.py`       |
| 7     | Post-processing    | `src/postprocessing/post_process.py`    |
| 7     | CAD export         | `src/export/export_cad.py`              |

## 5. Key Technical Details

- **SVG → YOLO conversion** handles two coordinate systems: structural elements
  (Wall/Door/Window/Stairs) use absolute SVG coordinates, while furniture
  (Toilet/Sink) use local coordinates that require applying the parent
  `matrix(...)` affine transform. Noise elements (direction arrows, labels) are
  filtered out of bounding boxes.
- **Preprocessing** uses aspect-ratio-preserving letterboxing to 1280×1280 —
  critical for floor plans with extreme aspect ratios, where a squash-resize
  would distort door arcs and wall angles.
- **Post-processing** merges fragmented wall segments, snaps wall endpoints at
  corners/T-junctions, aligns doors/windows into their host walls, and
  calibrates pixels-per-meter from detected door widths (~0.9 m standard).
- **DXF export** writes color-coded layers, real architectural symbols (door
  swing arcs, toilet bowls, staircase steps), and meter-scaled extents so the
  output opens correctly in AutoCAD.

## 6. Results (summary)

Final model: **YOLOv11l** — validation mAP@50 ≈ **0.838**, mAP@50-95 ≈ **0.571**.
See [EVALUATION.md](EVALUATION.md) for the full per-class breakdown.

## 7. Tech Stack

- **Deep learning:** PyTorch, Ultralytics YOLOv11
- **Computer vision:** OpenCV, Pillow
- **SVG parsing:** lxml, svgpathtools, ElementTree
- **CAD export:** ezdxf
- **Tooling:** ruff, pytest, Make

## 8. Team

 Tharun Umesh         
 Keshara Gunathilaka  
 Dilanka Hewage       
 Ashan Munasinghe     

## 9. Further Reading

- [QUICKSTART.md](QUICKSTART.md) — install and run the pipeline end to end
- [EVALUATION.md](EVALUATION.md) — metrics, methodology, per-class analysis
- [svg_findings.md](svg_findings.md) — how the CubiCasa SVG structure was decoded
