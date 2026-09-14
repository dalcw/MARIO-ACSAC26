# Training Guide

Run commands from the repository root after activating the `mario` conda environment. Training is optional because the artifact experiments use released checkpoints by default.

## Train MARIO on CIFAR-10

```bash
python artifact/scripts/train.py
```

Default configuration:

- Epochs: 30
- Batch size: 256
- Optimizer: Adam (`lr=1e-3`)
- Gradient clipping: max norm 5.0
- MARIO coefficients: `alpha=1e-3`, `beta=1e-3`
- Latent dimension: 128
- Dropout: 0.1

CIFAR-10 is downloaded automatically. The following files are written to `artifact/runs/cifar10_mario/`:

```text
best.pt
last.pt
train_log.csv
```

Use `python artifact/scripts/train.py --help` to view or override the defaults.

## Train the Reconstruction Attacker

Train a decoder from `x_pub` while keeping the trained MARIO model frozen:

```bash
python artifact/scripts/train_reconstruction.py \
  --checkpoint artifact/runs/cifar10_mario/best.pt \
  --save-images 10
```

Default configuration:

- Epochs: 20
- Batch size: 256
- Optimizer: Adam (`lr=1e-3`)
- Gradient clipping: max norm 5.0
- Loss: MSE

Outputs are written to `artifact/runs/cifar10_reconstruction/`:

```text
reconstruction_attacker.pt
reconstruction_attacker_last.pt
reconstruction_images/
```

## Retrain Evaluation Attackers

These optional commands train new attackers against frozen victim models. Run the corresponding evaluation with the generated checkpoint directory.

### Experiment 1: Reconstruction

```bash
./artifact/experiments/01_main_privacy_utility/train_attackers.sh
./claims/claim1/run.sh \
  --attacker-root "$(pwd)/artifact/generated_results/01_main_privacy_utility/scratch_checkpoints"
```

Configuration: Adam, `lr=1e-3`, MSE loss, and 30 epochs.

### Experiment 2: Property Inference

```bash
./artifact/experiments/02_property_inference/train_attackers.sh
./claims/claim2/run.sh \
  --attacker-root "$(pwd)/artifact/generated_results/02_property_inference/scratch_checkpoints"
```

Configuration: Adam, `lr=1e-3`, cross-entropy loss, and 10 epochs.

### Experiment 4: Ablation Attackers

```bash
./artifact/experiments/04_ablation/train_attackers.sh
./claims/claim4/run.sh \
  --attacker-root "$(pwd)/artifact/generated_results/04_ablation/scratch_checkpoints"
```

Configuration: batch size 32 and 10 epochs. Reconstruction and property inference use MSE and cross-entropy loss, respectively.

Scratch checkpoints and training summaries are written under `artifact/generated_results/.../scratch_checkpoints/`. Experiment 3 optimizes inputs directly, and Experiment 5 trains its reconstruction probes during evaluation.
