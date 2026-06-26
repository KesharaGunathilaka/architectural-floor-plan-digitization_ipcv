# Evaluation

How the model is evaluated, the metrics used, and the results achieved.

---

## 1. Metrics

The pipeline reports standard object-detection metrics via Ultralytics:

| Metric        | Meaning                                                              |
|---------------|---------------------------------------------------------------------|
| **Precision** | Of all predicted boxes, the fraction that are correct.              |
| **Recall**    | Of all ground-truth objects, the fraction that were detected.       |
| **mAP@50**    | Mean Average Precision at IoU ≥ 0.50 — the headline accuracy number.|
| **mAP@50-95** | Mean AP averaged over IoU 0.50→0.95 — rewards tight localization.   |
| **AP (per class)** | Average Precision computed separately for each of the 6 classes.|

IoU (Intersection over Union) measures how well a predicted box overlaps the
ground-truth box.

---

## 2. How to Run

```bash
make evaluate
# or
python src/evaluation/evaluate_test.py
```

This loads `models/final/best.pt` and evaluates it on the **test** split
(unseen during training) defined in `data/yolo_dataset_processed/dataset.yaml`.
It prints overall metrics plus a per-class AP@50 breakdown.

For a qualitative check, run visual inference (saves annotated images):

```bash
python src/evaluation/visual_inference_test.py   # → runs/detect/predict/
```

---

## 3. Data Split

The dataset is split **stratified by quality category** so every split keeps the
natural mix of `colorful` / `high_quality` / `high_quality_architectural`:

| Split | Ratio | Purpose                          |
|-------|-------|----------------------------------|
| train | 70%   | Model fitting                    |
| val   | 15%   | Tuning, early stopping           |
| test  | 15%   | Final unbiased evaluation        |

Splits are deterministic (`seed=42`), so results are reproducible.

---

## 4. Results

Two large backbones were trained and compared. **YOLOv11l is the final model.**

### Overall

| Model        | Precision | Recall | mAP@50 | mAP@50-95 |
|--------------|-----------|--------|--------|-----------|
| YOLOv8l      | 0.827     | 0.798  | 0.812  | 0.549     |
| **YOLOv11l** | —         | —      | **0.838** | **0.571** |

### Per-class AP@50 (YOLOv11l)

| Class     | AP@50  | Notes                                  |
|-----------|--------|----------------------------------------|
| Door      | 0.928  | Strongest — frequent, distinctive arc  |
| Window    | 0.906  | Strong                                 |
| Sink      | 0.892  | Strong despite being a rare class      |
| Wall      | 0.838  | Solid; high count but visually uniform |
| Staircase | 0.760  | Weaker — rare + visually variable      |
| Toilet    | 0.705  | Weakest — rare + small + diverse shapes|

> Metrics above are from the validation set during model development. Re-run
> `make evaluate` to produce the final **test-split** numbers from
> `models/final/best.pt`.

---

## 5. Targets vs. Achieved

The project set stretch targets of **mAP@50 ≥ 0.85** and **mAP@50-95 ≥ 0.60**.

| Target            | Status      | Gap     |
|-------------------|-------------|---------|
| mAP@50 ≥ 0.85     | Approaching | −0.012  |
| mAP@50-95 ≥ 0.60  | Approaching | −0.029  |

The two weakest classes (Toilet, Staircase) are the main drag on the average —
both are rare and visually diverse. The hyperparameter-tuning sweep
(`src/training/hyperparameter_tuning.py`) targets exactly these classes.

---

## 6. Class Imbalance & Mitigations

Annotation distribution across the full dataset:

| Class     | Share  |
|-----------|--------|
| Wall      | 53.7%  |
| Door      | 20.4%  |
| Window    | 18.0%  |
| Sink      | 3.2%   |
| Toilet    | 2.8%   |
| Staircase | 1.9%   |

Imbalance ratio Wall : Staircase ≈ **28 : 1**. Mitigations applied during
training:

1. **Class-loss weighting** (`cls=1.5`, up from the 0.5 default) — penalizes
   misclassification of rare classes more heavily.
2. **Mosaic augmentation** (`mosaic=1.0`) — combines 4 images, naturally
   over-representing images that contain rare classes.
3. **Copy-paste augmentation** (`copy_paste=0.3`) — pastes objects between
   images to synthesize more rare-class samples.
4. **Per-class recall monitoring** — flag Toilet/Sink recall if it drops below 0.50.

---

## 7. Hyperparameter Tuning

`make tune` runs a systematic One-Factor-At-A-Time (OFAT) sweep that isolates
the contribution of each lever, then combines the proven ones:

| Experiment | Lever                       | Hypothesis                              |
|------------|-----------------------------|-----------------------------------------|
| E01        | imgsz 640 → 1280            | Small objects need more pixels          |
| E02        | copy_paste 0.30 → 0.60      | More synthetic rare-class samples       |
| E03        | cls 1.5 → 2.5               | Sharper class discrimination            |
| E04        | box 7.5 → 10.0              | Tighter boxes → better mAP@50-95        |
| E05        | degrees = 90                | Orientation invariance                  |
| E06        | E01 + E02 + E03             | Combined minority-class fix             |
| E07        | E06 + E04 + E05             | Full combination                        |
| E08        | YOLO11x + E07 (conditional) | Architecture ceiling, only if E07 misses|

Results are written to `models/tuning_runs/tuning_results.csv` and
`tuning_summary.txt`, ranked by mAP@50 with target hit/miss flags.

---

## 8. Reproducing the Evaluation

```bash
# 1. Regenerate the deterministic split (seed=42)
make dataset-stats && make select-data

# 2. Convert + preprocess
make convert && make preprocess

# 3. Train (or drop in pre-trained weights at models/final/best.pt)
make train
cp runs/detect/models/11l_final/weights/best.pt models/final/best.pt

# 4. Evaluate on the held-out test split
make evaluate
```
