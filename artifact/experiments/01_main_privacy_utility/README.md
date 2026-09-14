# Experiment 1: Main Privacy-Utility Evaluation

This experiment evaluates the task accuracy and supervised reconstruction attack
(PSNR, SSIM, and AlexNet LPIPS) on the test set. By default, it evaluates all six
methods on CIFAR-10 and stores quantitative and qualitative outputs under
`claims/claim1/results/`.

```bash
./claims/claim1/run.sh
```

Use `--max-samples N` only when a resource-constrained evaluator needs a scaled
run. Omitting it evaluates all 10,000 CIFAR-10 test examples.

The release attacker checkpoints are used by default. To train only the reconstruction attackers from scratch against the fixed defense checkpoints and then evaluate them, run:

```bash
./artifact/experiments/01_main_privacy_utility/train_attackers.sh
./claims/claim1/run.sh \
  --attacker-root "$(pwd)/artifact/generated_results/01_main_privacy_utility/scratch_checkpoints"
```

The paper protocol uses Adam with learning rate `1e-3`, MSE loss, and 30 epochs for CIFAR-10 (10 epochs for CelebA and NIH Chest X-ray). Scratch checkpoints and their training summary are saved under the selected `--output-root`.
