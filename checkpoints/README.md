# Checkpoints

## Download

Download the checkpoint archives listed in the top-level [`README.md`](../README.md).

## Extract

Extract each downloaded archive into `checkpoints/release/` from the repository root:

```bash
mkdir -p checkpoints/release
tar -xzf /path/to/mario-cifar10-checkpoints.tar.gz -C checkpoints/release
tar -xzf /path/to/mario-celeba-checkpoints.tar.gz -C checkpoints/release
tar -xzf /path/to/mario-nih-checkpoints.tar.gz -C checkpoints/release
```

The resulting directory structure is:

```text
checkpoints/release/
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
python scripts/verify_checkpoint_hashes.py
python scripts/check_setup.py
```
