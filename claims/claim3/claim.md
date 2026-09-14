# Claim 3: Likelihood-Maximization Resistance

MARIO resists direct likelihood-maximization reconstruction on CIFAR-10.

- **Paper reference:** Figure 6
- **Run:** `./claims/claim3/run.sh`
- **Observed result:** `claims/claim3/results/cifar10/summary.csv`
- **Expected result:** `claims/claim3/expected/metrics.csv`
- **Validation:** Confirm that white-box input optimization produces low-fidelity MARIO reconstructions according to PSNR, SSIM, and LPIPS.
