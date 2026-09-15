# Checkpoints

## Download

Download the checkpoint archives listed in the top-level [`README.md`](../../README.md).

## Extract

Save the downloaded archives under `artifact/checkpoint_archives/`. From the
repository root, extract them into `artifact/checkpoints/release/`:

```bash
mkdir -p artifact/checkpoint_archives artifact/checkpoints/release
tar -xzf artifact/checkpoint_archives/mario-cifar10-checkpoints.tar.gz -C artifact/checkpoints/release
tar -xzf artifact/checkpoint_archives/mario-celeba-checkpoints.tar.gz -C artifact/checkpoints/release
tar -xzf artifact/checkpoint_archives/mario-nih-checkpoints.tar.gz -C artifact/checkpoints/release
```

The resulting directory structure is:

```text
artifact/checkpoints/release/
|-- cifar10/
|-- celeba/
|-- nih_chest_xray/
|-- ablation/
`-- latent_decomposition/
```

Only the directories for the experiments being evaluated are required.

## Verify

After extracting all checkpoint archives, verify the files and environment:

```bash
python artifact/scripts/verify_checkpoint_hashes.py
python artifact/scripts/check_setup.py
```
