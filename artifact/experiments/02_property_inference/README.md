# Experiment 2: Property-Inference Attack

This experiment evaluates pretrained binary attackers against the two sensitive
attributes used in the paper. The default run evaluates Male and Young prediction
on all 39,829 CelebA test images for every defense.

```bash
./claims/claim2/run.sh
```

NIH Chest X-ray evaluation is selected with `--dataset nih_chest_xray`. Use
`--max-samples N` only for a resource-constrained scaled run.

The release attackers are used by default. To train only the two binary property attackers from scratch against each fixed defense model and evaluate the resulting checkpoints, run:

```bash
./artifact/experiments/02_property_inference/train_attackers.sh
./claims/claim2/run.sh \
  --attacker-root "$(pwd)/artifact/generated_results/02_property_inference/scratch_checkpoints"
```

Training uses Adam with learning rate `1e-3`, cross-entropy loss, and 10 epochs. No Vanilla, baseline, or MARIO model parameter is updated.
