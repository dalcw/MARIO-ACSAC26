# Infrastructure Constraints

- No author-operated private infrastructure is required for the default evaluation.
- Run one experiment process on one CUDA-capable GPU.
- Checkpoint-based Claims 1–5 are configured for approximately 24 GB of GPU memory using the documented default batch sizes.
- CPU execution is supported by PyTorch but is impractical for full-scale evaluation.
- Allow at least 20 GB of additional free disk space, excluding the space required for datasets and downloaded checkpoint archives.