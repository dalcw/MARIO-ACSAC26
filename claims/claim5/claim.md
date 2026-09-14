# Claim 5: Latent Information Separation

The private latent is more reconstructive than the server-exposed public latent under matched probes.

- **Paper reference:** Figure 11
- **Run:** `./claims/claim5/run.sh`
- **Observed result:** `claims/claim5/results/metrics.csv`
- **Expected result:** `claims/claim5/expected/metrics.csv`
- **Validation:** Confirm that `z_priv` has higher PSNR and SSIM and lower LPIPS than `z_pub` under matched decoder capacity and training.
