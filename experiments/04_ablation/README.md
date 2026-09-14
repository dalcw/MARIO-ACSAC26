# Experiment 4: Stage-Wise Ablation

This experiment reproduces the CelebA comparison among Vanilla, channel
obfuscation only, latent decomposition only, variational sampling only, and the
full MARIO model. It reports Smiling task accuracy, AlexNet LPIPS, and Male
property-inference accuracy as in Table 3. PSNR and SSIM are retained as
supplementary outputs.

```bash
./experiments/04_ablation/run.sh
```

Historical checkpoints contained parameters for disabled branches. The release
checkpoints retain only tensors reachable by each evaluated variant; the
extraction and exact tensor checks are implemented in
`scripts/prepare_ablation_checkpoints.py`.

The release attackers are used by default. To optionally train the reconstruction and Male-property attackers from scratch while all five ablation models remain frozen, run:

```bash
./experiments/04_ablation/train_attackers.sh
./experiments/04_ablation/run.sh --attacker-root generated_results/04_ablation/scratch_checkpoints
```

Training uses Adam with learning rate `1e-3`, MSE for reconstruction, and cross-entropy for property inference for 10 epochs. The default training batch size is 32 to keep the two 224x224-pixel attackers within practical GPU memory.
