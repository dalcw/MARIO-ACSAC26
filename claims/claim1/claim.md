# Claim 1: Reconstruction Privacy and Task Utility

MARIO lowers supervised reconstruction fidelity while retaining competitive task accuracy.

- **Paper reference:** Figures 4 and 9
- **Run:** `./claims/claim1/run.sh`
- **Observed result:** `claims/claim1/results/metrics.csv`
- **Expected result:** `claims/claim1/expected/metrics.csv`
- **Validation:** Compare all rows and confirm that MARIO has substantially lower PSNR and SSIM and higher LPIPS than Vanilla while retaining competitive accuracy.
