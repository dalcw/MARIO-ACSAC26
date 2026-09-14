# Claim-to-Experiment Mapping

## Claim 1: Reconstruction Privacy and Task Utility

MARIO reduces the fidelity of supervised input reconstruction while retaining
competitive task accuracy. Run `experiments/01_main_privacy_utility/run.sh` and
compare `generated_results/01_main_privacy_utility/metrics.csv` with
`expected_results/01_main_privacy_utility.csv`.

## Claim 2: Property-Inference Resistance

The server-exposed MARIO representation contains less exploitable sensitive
attribute information. Run `experiments/02_property_inference/run.sh` and compare
the Male and Young accuracies with `expected_results/02_property_inference.csv`.

## Claim 3: Likelihood-Maximization Resistance

Direct input optimization against the exposed representation produces low-fidelity
MARIO reconstructions. Run `experiments/03_likelihood_maximization/run.sh` and
compare the three reconstruction metrics with
`expected_results/03_likelihood_maximization.csv`.

## Claim 4: Complementary Stages

Channel obfuscation, latent decomposition, and variational sampling address
different leakage channels, and their combination gives the best overall
privacy-utility balance. Run `experiments/04_ablation/run.sh` and compare the
result with `expected_results/04_ablation.csv`.

## Claim 5: Latent Information Separation

The private latent supports more faithful reconstruction than the public latent
under matched decoder capacity and training. Run
`experiments/05_latent_decomposition/run.sh` and verify that `z_priv` has higher
PSNR/SSIM and lower LPIPS than `z_pub`, as summarized in
`expected_results/05_latent_decomposition.csv`.
