# DP
# Brain Tumor Segmentation: U-Net, Attention U-Net, ResUNet and DeepLabV3+ under Adadelta

A controlled comparison of four 2D segmentation architectures for multi-region brain tumor segmentation
(Whole Tumor **WT**, Tumor Core **TC**, Enhancing Tumor **ET**), all trained with the **Adadelta** optimizer
and evaluated over three random seeds.

> **Status:** research code / work in progress. Items marked `TODO` must be filled in before publishing.

---

## Models

| Model | Short description |
|---|---|
| **Basic 2D U-Net** | Standard encoder–decoder with skip connections |
| **Attention U-Net** | U-Net with attention gates on the skip connections |
| **ResUNet** | U-Net with residual blocks |
| **DeepLabV3+** | Atrous spatial pyramid pooling encoder with a light decoder |

---

## Experimental setup

| Item | Value |
|---|---|
| Dataset | `TODO` (e.g. BraTS year / source, modalities used) |
| Split | Patient-level, 150 train / 50 val / 50 test patients |
| Slices per split (seed 42) | 11,070 / 3,725 / 3,573 |
| Task | 2D slice-wise segmentation of WT, TC, ET |
| Loss | Dice loss |
| Optimizer | Adadelta, initial `lr = 1e-3` (`TODO`: confirm other arguments, e.g. `rho`, `eps`, weight decay) |
| LR schedule | Reduce-on-plateau, learning rate halved on plateau |
| Max epochs | 200 |
| Early stopping | Patience 15 epochs on validation mean Dice |
| Model selection | Best validation mean Dice checkpoint |
| Seeds | 42, 43, 44 (the patient split changes with the seed) |
| Batch size / image size / augmentation | `TODO` |
| Hardware | `TODO` (GPU model) |

Test metrics are computed on the held-out test patients with the checkpoint chosen on validation data only.

---

## Results (mean ± std over 3 seeds)

### Segmentation accuracy

| Metric | U-Net | Attention U-Net | DeepLabV3+ | ResUNet |
|---|---|---|---|---|
| Mean Dice | 0.8251 ± 0.0254 | 0.8331 ± 0.0071 | 0.8395 ± 0.0032 | **0.8509 ± 0.0081** |
| WT Dice | 0.8906 ± 0.0059 | 0.8978 ± 0.0051 | 0.8968 ± 0.0078 | **0.9027 ± 0.0057** |
| TC Dice | 0.8318 ± 0.0340 | 0.8332 ± 0.0154 | 0.8515 ± 0.0138 | **0.8538 ± 0.0162** |
| ET Dice | 0.7529 ± 0.0393 | 0.7684 ± 0.0165 | 0.7701 ± 0.0118 | **0.7963 ± 0.0141** |
| Mean IoU | 0.7068 ± 0.0345 | 0.7176 ± 0.0095 | 0.7270 ± 0.0040 | **0.7431 ± 0.0116** |
| Precision | 0.8292 ± 0.0192 | 0.8440 ± 0.0170 | 0.8451 ± 0.0055 | **0.8641 ± 0.0084** |
| Recall | 0.8216 ± 0.0332 | 0.8234 ± 0.0244 | 0.8342 ± 0.0100 | **0.8385 ± 0.0158** |
| WT HD95 (lower is better) | 6.16 ± 0.71 | **5.28 ± 0.10** | 5.44 ± 0.02 | 5.95 ± 0.54 |
| TC HD95 | 4.13 ± 0.62 | 3.79 ± 0.10 | **3.41 ± 0.15** | 3.54 ± 0.36 |
| ET HD95 | 4.23 ± 0.70 | 3.78 ± 0.35 | 3.63 ± 0.28 | **3.38 ± 0.27** |

### Efficiency

| Metric | U-Net | Attention U-Net | DeepLabV3+ | ResUNet |
|---|---|---|---|---|
| Parameters (M) | 7.76 | 7.85 | **2.56** | 8.12 |
| FLOPs (G) | 6.05 | 6.18 | **5.12** | 6.39 |
| GPU memory (MB) | 168.8 | 171.1 | **66.5** | 175.7 |
| Inference time (ms) | **3.72** | 3.95 | 5.40 | 3.82 |
| Training time (s) | 7332 | 7925 | **5287** | 8205 |
| Convergence epoch | 198 | 199 | **163** | 188 |

Best value per row in **bold**. `TODO`: state how inference time, FLOPs and GPU memory were measured (input size, batch size, device).

### Per-seed results

Per-seed values are available in the `results/` folder (`TODO`: add CSV files).

---

## Key findings

1. **ResUNet** achieved the highest mean Dice, IoU, precision and recall, and was ahead of DeepLabV3+ on every seed. The largest gain is on the enhancing tumor (ET).
2. **DeepLabV3+** is the most efficient model (about 3x fewer parameters and about 2.5x less GPU memory than the others) and the most stable across seeds, at a small accuracy cost compared with ResUNet.
3. **Attention U-Net** improves slightly over the basic U-Net, mainly in ET Dice, HD95 and variance across seeds.
4. The **basic U-Net** has the lowest mean Dice and the highest variance across seeds.
5. ET is the hardest region for every model; small or scattered tumors in some slices remain difficult.

---

## Limitations

Please read these before citing the numbers.

- **Only 3 seeds.** In several cases the gap between models is smaller than the seed-to-seed standard deviation (for example U-Net vs. Attention U-Net). No statistical significance is claimed.
- **Training was cut off for several runs.** U-Net, Attention U-Net and ResUNet reached the 200-epoch limit on most seeds, and validation Dice was still rising. Model rankings may partly reflect the training budget rather than the architecture alone.
- **Adadelta was run with `lr = 1e-3`.** The PyTorch default for Adadelta is `lr = 1.0`, so the optimizer may have been under-scaled, which would explain the long slow start visible in the training curves. Results for other optimizers or learning rates may differ. `TODO`: confirm and, if applicable, add a learning-rate sensitivity study.
- **Single dataset, single optimizer, 2D slices only.** Conclusions may not transfer to 3D models, other datasets or other optimizers.
- Splits differ between seeds, so the standard deviation mixes seed effects with split effects.

---

## Repository structure

`TODO`: adjust to your actual layout.

```
.
├── configs/            # experiment configs
├── models/             # unet.py, attention_unet.py, resunet.py, deeplabv3plus.py
├── data/               # dataset loading / preprocessing
├── train.py            # training loop
├── evaluate.py         # test-set evaluation (Dice, IoU, precision, recall, HD95)
├── results/            # per-seed CSVs and summary files
├── figures/            # training curves and qualitative examples
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # TODO: add requirements.txt
```

## Usage

`TODO`: replace with your real commands and arguments.

```bash
# Train one model with one seed
python train.py --model resunet --optimizer adadelta --lr 1e-3 --seed 42 --epochs 200

# Evaluate on the test split
python evaluate.py --model resunet --checkpoint checkpoints/resunet_seed42.pt
```

Available `--model` values: `unet`, `attention_unet`, `resunet`, `deeplabv3plus`.

---

## Reproducibility notes

- Fixed seeds: 42, 43, 44.
- Patient-level split, so slices from one patient never appear in more than one of train, validation and test.
- Checkpoint selection uses validation data only.
- `TODO`: library versions, CUDA version, and whether deterministic algorithms were enabled.

---



## License

`TODO` (for example MIT). Check the license terms of the dataset you use separately.

## Acknowledgements

`TODO`: dataset providers, compute resources, and the original papers for U-Net, Attention U-Net, ResUNet and DeepLabV3+.
