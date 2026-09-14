# Claim-to-Experiment Mapping

## Claim 1: Reconstruction Privacy and Task Utility

MARIO reduces the fidelity of supervised input reconstruction while retaining
competitive task accuracy. Run `claims/claim1/run.sh` and compare
`claims/claim1/results/metrics.csv` with
`claims/claim1/expected/metrics.csv`.

## Claim 2: Property-Inference Resistance

The server-exposed MARIO representation contains less exploitable sensitive
attribute information. Run `claims/claim2/run.sh` and compare the Male and Young
accuracies with `claims/claim2/expected/metrics.csv`.

## Claim 3: Likelihood-Maximization Resistance

Direct input optimization against the exposed representation produces low-fidelity
MARIO reconstructions. Run `claims/claim3/run.sh` and compare the three
reconstruction metrics with
`claims/claim3/expected/metrics.csv`.

## Claim 4: Complementary Stages

Channel obfuscation, latent decomposition, and variational sampling address
different leakage channels, and their combination gives the best overall
privacy-utility balance. Run `claims/claim4/run.sh` and compare the result with
`claims/claim4/expected/metrics.csv`.

## Claim 5: Latent Information Separation

The private latent supports more faithful reconstruction than the public latent
under matched decoder capacity and training. Run
`claims/claim5/run.sh` and verify that `z_priv` has higher
PSNR/SSIM and lower LPIPS than `z_pub`, as summarized in
`claims/claim5/expected/metrics.csv`.
