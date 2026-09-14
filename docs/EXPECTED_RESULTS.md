# Expected Results

Each `claims/claimN/expected/metrics.csv` records values reported in the paper or
retained experiment logs. Full test-set evaluation should closely match Claims 1 and 2. Results
with stochastic defenses and newly optimized models can vary with hardware and
random-number implementation; compare the direction and rounded values rather
than requiring bitwise-identical floating-point output.

Experiment 3 optimizes only 16 images and is consequently the most seed-sensitive.
Experiment 5 retrains two matched reconstruction probes, so its expected result is
that `z_priv` reconstructs better than `z_pub` (higher PSNR/SSIM and lower LPIPS),
with the reported values serving as reference points.
