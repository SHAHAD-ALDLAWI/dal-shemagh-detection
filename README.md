# 🧣 Dal Shemagh Detection Challenge

Object detection model that detects **heads** and **shemaghs** in images and decides whether the shemagh is worn in the right place — built for the **Dal Shemagh Detection Challenge** on Kaggle.

**Result:** ranked **63 out of 140+** participants.

---

## 📌 Task

For every test image, the model must:
1. Detect heads (class `0`) and shemaghs (class `1`) with bounding boxes.
2. Predict `right_place` — `1` if a shemagh is correctly positioned on a head, otherwise `0`.

| Split | Images |
|---|---|
| Train (labeled) | 651 |
| Test (unlabeled) | 842 |

Class distribution in training labels: **446 heads**, **165 shemaghs**, **203 background images** (no objects) — an imbalanced dataset with a large share of negatives.

## 🛠️ Approach

- **Framework:** [Ultralytics YOLO](https://docs.ultralytics.com/) (transfer learning from COCO-pretrained weights)
- **Right-place logic:** compute IoU between every detected head and shemagh box; if any pair overlaps above a threshold → `right_place = 1`.
- **Hardware:** Kaggle, NVIDIA Tesla T4

## 🔁 Experiments

| # | Model | Epochs | Key changes | IoU threshold | Images flagged `right_place=1` |
|---|---|---|---|---|---|
| 1 | YOLOv8n | 100 | Baseline, 80/20 train/val split, AdamW + cosine LR | 0.3 | 31 |
| 2 | YOLO11n | 150 | Newer architecture, trained on all 651 images | 0.1 | 83 |
| 3 | YOLO11s | 200 | Larger model, lower LR (`lr0=0.001`), conf 0.15 | 0.1 | 122 |

**Baseline validation (held-out 20%):** mAP50 = **0.932**, mAP50-95 = **0.694** (head 0.795, shemagh 0.593).

## 💡 Lessons learned

- **Validation must stay held out.** In experiments 2–3 the validation set pointed at the training images, so their reported mAP (0.97+) reflects memorization, not generalization. The baseline's 80/20 split is the only trustworthy offline metric; next step is k-fold CV to use all data *and* measure honestly.
- **The decision rule matters as much as the detector.** Lowering the IoU threshold from 0.3 → 0.1 nearly tripled positive predictions — the heuristic should be tuned against labeled data rather than set by hand.
- **Shemagh is the harder class** (mAP50-95 0.59 vs 0.80 for heads): fewer examples and highly variable shape/folds.

## 🚀 Future improvements

- Stratified K-fold validation and tuning the IoU / confidence thresholds on validation data
- Replace IoU with a containment/position rule (shemagh box covering the upper part of the head box)
- Class-balanced sampling or extra
