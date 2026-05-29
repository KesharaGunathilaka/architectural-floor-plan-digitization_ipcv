# Automated Digitization of Architectural Floor Plans

## Problem
Millions of architectural floor plans exist only as static PDFs or scanned paper
blueprints. Computers cannot read the spatial data in these images. Manual
re-digitization is slow, expensive, and error-prone. This project develops a
Computer Vision pipeline using YOLOv8 to automatically detect and extract
architectural symbols (doors, windows, walls, staircases, toilets, sinks) from
floor plan images and export them as structured CAD-compatible vector files.

## Pipeline Overview
Raw Floor Plans → Subset Selection → SVG→YOLO Conversion
→ Image Preprocessing → YOLOv8 Training
→ Evaluation → JSON/DXF Export

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/tharUmesh/architectural-floor-plan-digitization.git
cd architectural-floor-plan-digitization
make setup
```

### 2. Configure Paths
```bash
# Edit .env with your cubicasa5k dataset location
nano .env
```

### 3. Run the Pipeline
```bash
make select-data     # Select 500-image stratified subset
make audit-svg       # Explore SVG structure
make convert         # Convert annotations to YOLO format
make verify          # Visually verify annotations
make preprocess      # Preprocess images
make train           # Train YOLOv8s (run on Ubuntu GPU machine)
make evaluate        # Evaluate on test set
make postprocess     # Export to JSON/DXF
```

## Dataset
[CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k) — 5,000 annotated
residential floor plans. Not included in this repository. Download separately
and set `CUBICASA_ROOT` in your `.env`.

## Requirements
- Python 3.10+
- CUDA-compatible GPU recommended for training (tested on RTX 5090)
- See `requirements.txt` for full dependency list

## Project Structure
configs/        YAML configuration files
data/           Dataset (gitignored) and split records
docs/           Research notes and references
notebooks/      Exploration and verification notebooks
src/            All source code
data/         Dataset selection and annotation conversion
preprocessing/Image preprocessing pipeline
training/     Model training
evaluation/   Metrics and visualization
postprocessing/Vectorization and export
utils/        Shared utilities
tests/          Unit tests

## Results
*To be updated after training.*

| Model | mAP@50 | mAP@50-95 | Inference (ms) |
|-------|--------|-----------|----------------|
| YOLOv8s | — | — | — |
| YOLOv11s | — | — | — |