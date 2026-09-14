# Experiment 5: Latent Decomposition

This experiment trains capacity-matched image decoders on the frozen `z_priv` and
`z_pub` representations and evaluates both on the same 500 CIFAR-10 test images.
The default setting uses the first 2,000 training images and trains each decoder
for 50 epochs, matching Figure 11.

```bash
./claims/claim5/run.sh
```

The trained decoder checkpoints are saved under
`claims/claim5/results/decoders/`. A repeated evaluation can
reuse them by adding `--reuse-decoders`.
