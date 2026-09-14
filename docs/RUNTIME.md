# Runtime and Resource Guidance

## Reference Environment

- OS: Ubuntu 24.04.4 LTS
- CPU: 2x Intel Xeon Gold 6326
- Logical CPUs: 64
- System memory: 125 GiB
- GPU: 2x NVIDIA RTX PRO 6000 Blackwell Max-Q
- GPU memory: 96 GB per GPU
- NVIDIA driver: 580.173.02
- Python: 3.14.4
- PyTorch: 2.11.0+cu130
- torchvision: 0.26.0+cu130
- CUDA runtime: 13.0

## 24 GB Evaluation Support

- Supported scope: checkpoint-based Experiments 1-5
- Execution mode: one experiment process on one GPU
- Batch policy: smaller batches for 224x224 inputs
- Dataset scope: full test set remains unchanged
- Primary trade-off: lower memory usage with longer evaluation
- CPU execution: supported but impractical for full runs
- Scratch training: not covered by the 24 GB evaluation guarantee

| Experiment | Default batch | First value to try after OOM |
| --- | ---: | ---: |
| 1 (CIFAR-10) | 256 | 128 |
| 2 (CelebA property inference) | 8 | 4 |
| 3 (CIFAR-10 LMA) | 4 | 2 |
| 4 (CelebA ablation) | 32 | 16 |
| 5 (CIFAR-10 latent decomposition) | 32 | 16 |

## Measured GPU Memory

- Measurement device: GPU 1 on the reference system
- CelebA DISCO, batch 32: approximately 26.1 GiB peak allocated/reserved
- CelebA DISCO, batch 16: approximately 13.1 GiB peak allocated/reserved
- CelebA full-MARIO ablation, batch 32: approximately 4.9 GiB peak allocated and 8.1 GiB peak reserved
- Variability: exact peaks depend on the GPU, CUDA, cuDNN, and PyTorch versions

## Reducing Batch Size

Every `run.sh` accepts a `--batch-size` argument. For example:

```bash
./experiments/[experiment name]/run.sh --batch-size 16
```

Replace `[experiment name]` with the target experiment directory. If the process still runs out of memory, halve the value again until it fits. This changes memory use and execution time without reducing the evaluation dataset. Experiments 3 and 5 include stochastic optimization, so their exact values may vary slightly when the batch size changes.
