# Quick Start

Get the floor-plan digitization pipeline running from a clean checkout.

---

## Prerequisites

- **Python 3.10+**
- **CUDA-compatible GPU** strongly recommended for training (development used an
  RTX 5090, 32 GB VRAM). Inference and export run fine on CPU.
- **CubiCasa5k dataset** — [download separately](https://github.com/CubiCasa/CubiCasa5k)
  (not included in this repo).
- **GNU Make** (optional but recommended — every stage has a `make` target).

---

## 1. Clone and Install

```bash
git clone https://github.com/tharUmesh/architectural-floor-plan-digitization.git
cd architectural-floor-plan-digitization

make setup            # creates .env from .env.example + installs dev deps
```

`make setup` is equivalent to:

```bash
pip install -r requirements-dev.txt    # prod deps + ruff/pytest
pip install -e .                        # install the package itself
```

> **Production-only install:** `make install` (uses `requirements.txt`, no test/lint tooling).

---

## 2. Configure Paths

Edit `.env` and set the absolute path to your downloaded dataset:

```bash
CUBICASA_ROOT=/absolute/path/to/cubicasa5k
DEVICE=cuda:0          # or "cpu"
```

`.env` is gitignored. Use `.env.example` as the template.

---

## 3. Run the Pipeline

Each command is one phase. Run them in order — later phases depend on earlier
outputs.

```bash
# ── Phase 1: Data preparation ───────────────────────────────
make audit-svg       # Inspect SVG structure (writes docs/svg_audit_report.txt)
make dataset-stats   # Scan dataset, compute stats (REQUIRED before selection)
make select-data     # Stratified subset + train/val/test split (seed=42)

# ── Phase 2: Annotation conversion ──────────────────────────
make convert         # SVG annotations → YOLO .txt labels + copy images
make verify          # Draw boxes on samples → data/verify/ (visual sanity check)

# ── Phase 3: Preprocessing ──────────────────────────────────
make preprocess      # Letterbox images to 1280×1280, recompute boxes

# ── Phase 5: Training ───────────────────────────────────────
make train           # Train YOLOv11l (GPU). Writes runs/detect/models/11l_final/
# make tune          # (optional) systematic hyperparameter sweep

# ── Phase 6 & 7: Evaluation + Export ────────────────────────
make evaluate        # Test-set metrics + per-class AP
make postprocess     # Run inference + geometry refine → JSON + DXF
```

> **Note:** the split JSONs under `data/splits/` are gitignored, machine-local
> artifacts (they embed absolute dataset paths). Regenerate them with
> `make dataset-stats && make select-data` on any new machine — `seed=42` makes
> the selection identical for the same dataset.

---

## 4. Promote the Trained Model

After training, the best weights land in
`runs/detect/models/11l_final/weights/best.pt`. The evaluation, post-processing,
and export scripts all read from a single canonical location — copy the weights
there:

```bash
mkdir -p models/final
cp runs/detect/models/11l_final/weights/best.pt models/final/best.pt
```

`models/final/best.pt` is the path every downstream script uses.

---

## 5. Convert a Single Floor Plan → CAD

Once `models/final/best.pt` exists, run the full image-to-CAD pipeline on any
floor-plan image:

```bash
python src/export/export_cad.py path/to/floorplan.png
```

Outputs land in `runs/exports/<image_name>/`:
- `<name>.json` — structured list of detected elements with real-world coords
- `<name>.dxf`  — layered, meter-scaled AutoCAD drawing

Running it with no argument processes a default test image.

---

## 6. Quality Checks

```bash
make test            # pytest with coverage
make lint            # ruff
make clean           # remove __pycache__, .pytest_cache, coverage artifacts
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUBICASA_ROOT not set` | Edit `.env` and set the dataset path. |
| `dataset_stats.json not found` | Run `make dataset-stats` before `make select-data`. |
| `CUDA is not available` | Reinstall torch with the CUDA wheel (see the command printed by `train_yolo.py`), or set `DEVICE=cpu` for inference. |
| Empty/blank DXF | Confirm `models/final/best.pt` exists and the input is a real floor plan. |
| No annotations extracted | Verify `CUBICASA_ROOT` points at the folder containing `colorful/`, `high_quality/`, `high_quality_architectural/`. |
