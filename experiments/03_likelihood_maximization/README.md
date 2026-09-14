# Experiment 3: Likelihood-Maximization Attack

This experiment optimizes candidate inputs against the exposed representation.
The default command reproduces the CIFAR-10 setting used in the paper: 16 test
images, 2,000 Adam steps, learning rate 0.05, MSE feature matching, total-variation
weight 1e-4, and L2 weight 1e-5.

```bash
./experiments/03_likelihood_maximization/run.sh
```

Results and reconstructed images are written to
`generated_results/03_likelihood_maximization/`.
