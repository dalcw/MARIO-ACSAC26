# *MARIO*: Multi-stage Adapter for Reducing Privacy Leakage in Split Learning through Representation Exposure Control

![Status](https://img.shields.io/badge/Status-ACSAC%202026%20Accepted-brightgreen)
![Python](https://img.shields.io/badge/Python-3.14.4-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900)



> ***MARIO*: Multi-stage adapter for reducing privacy leakage in split learning through representation exposure control**

> Seongsu Moon, Taejune Park<br>
> In ACSAC 2026<br>

> **(Abstract)** Split learning enables collaborative training without sharing raw inputs, but intermediate representations (i.e., smashed data) can still leak structural and semantic information about the original input. Existing privacy mechanisms attempt to mitigate this leakage through perturbation, regularization, or suppression of representations, but often degrade task utility or fail to sufficiently reduce exploitable information. This paper presents MARIO, a lightweight client-side adapter for representation-level exposure control in split learning. Rather than perturbing smashed data as a whole, MARIO regulates what information remains exposed to the server through a three-stage pipeline consisting of channel obfuscation, latent-space decomposition, and variational sampling. Together, these mechanisms suppress reconstruction-related cues, reduce unnecessary information exposure, and weaken stable input--representation associations exploitable for reconstruction and property inference attacks. Evaluations on natural, facial, and medical image datasets show that MARIO consistently reduces reconstruction and property-inference leakage while maintaining competitive task accuracy and practical runtime overhead. These results suggest that privacy in split learning can be improved by explicitly controlling information exposure through intermediate representations.



## Repository Layout

```text
.
|-- checkpoints/
|   |-- README.md
|   `-- release/                  # downloaded weights; ignored by Git
|-- docs/
|   |-- CLAIMS.md
|   |-- DATASETS.md
|   |-- IMPLEMENTATION_NOTES.md
|   |-- PUBLIC_RELEASE.md
|   |-- RUNTIME.md
|   `-- TRAINING.md
|-- expected_results/             # paper/reference values
|-- experiments/
|   |-- 01_main_privacy_utility/
|   |-- 02_property_inference/
|   |-- 03_likelihood_maximization/
|   |-- 04_ablation/
|   `-- 05_latent_decomposition/
|-- mario/                        # models, attacks, data, and metrics
|-- scripts/                      # core training/inference and setup checks
|-- MARIO_CIFAR10_artifact.ipynb
|-- environment.yml
|-- metadata.toml
|-- requirements.txt
`-- setup_conda_env.sh
```

Each experiment has its own `run.sh`; there is intentionally no aggregate runner.

## Setup

Create or update the conda environment, then activate `mario`:

```bash
./setup_conda_env.sh
conda activate mario
```

**Reference environment**

- OS: Ubuntu 24.04.4 LTS
- CPU: 2x Intel Xeon Gold 6326 (64 logical CPUs total)
- System memory: 125 GiB
- GPU: 2x NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB each)
- NVIDIA driver: 580.173.02
- Python: 3.14.4
- PyTorch / torchvision: 2.11.0+cu130 / 0.26.0+cu130
- CUDA runtime: 13.0

**24 GB evaluation support**

- Scope: checkpoint-based Experiments 1-5
- Configuration: memory-conscious batches supplied by each `run.sh`
- Exclusion: scratch training may require a smaller batch or more memory
- Details: [`docs/RUNTIME.md`](docs/RUNTIME.md)

**CUDA out-of-memory**

Every experiment launcher accepts `--batch-size`. If an OOM error occurs, rerun the command with half the current batch size:

```bash
./experiments/[experiment name]/run.sh --batch-size 16
```

Replace `[experiment name]` with the target experiment directory. Continue halving the value if necessary. Reducing the batch size does not reduce the number of evaluated samples. See [`docs/RUNTIME.md`](docs/RUNTIME.md) for per-experiment values.

## Checkpoints

Release checkpoints are hosted externally because the complete weights are about
12 GB. Download only the archives needed for the claims being evaluated.

| Scope | Download | SHA-256 |
| --- | --- | --- |
| CIFAR-10 and latent decomposition | [Download](https://drive.google.com/file/d/1ztoRbk9Uj-6X84bIbPdyHOmM1Xbhh1Dz/view?usp=sharing) | `4d723a12ef1495e10bb8451a190c9c59571c2a536529a3b3e077981dfcecc9f2` |
| CelebA and ablation | [Download](https://drive.google.com/file/d/1bugPnLma3U6bQ2s2HxqawJiEIX6mQS82/view?usp=sharing) | `2f75fcbc81719acd6f93c680f820059a41f116bfbe9e0f90568549d912bb5e92` |
| NIH Chest X-ray | [Download](https://drive.google.com/file/d/1-ODFUJp8bt696dfhDHLkMKO1HrFaisXB/view?usp=sharing) | `45be6f33f58c7276d300f71c6927a175106e92645bd09c315717b64d7d68585d` |

Follow [`checkpoints/README.md`](checkpoints/README.md) for archive extraction, the required directory layout, and checksums. Then verify the weights and core environment:

```bash
python scripts/verify_checkpoint_hashes.py
python scripts/check_setup.py
```

## Datasets

Every experiment uses repository-local `data/` by default. CIFAR-10 is downloaded automatically; CelebA and NIH ChestXray14 require manual download. The official pages are listed for dataset provenance and access terms, while the Kaggle distributions identify the prepared files used by this artifact.

| Dataset | Official page | Download used | Required local path |
| --- | --- | --- | --- |
| CIFAR-10 | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | Automatic via torchvision | `data/cifar-10-batches-py/` |
| CelebA | [CelebA project](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | [Kaggle CSV package](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) | `data/celeba/` |
| NIH ChestXray14 | [NIH release](https://nihcc.app.box.com/v/ChestXray-NIHCC) | [Kaggle 224x224 package](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized) | `data/nih_chest_xray/` |

We use the prepared Kaggle distributions to make the artifact setup reproducible and to avoid requiring evaluators to repeat alignment, resizing, and metadata-format conversion. See [`docs/DATASETS.md`](docs/DATASETS.md) for the exact extraction layout, required metadata files, labels, splits, and custom `--data-root` usage.

## Artifact Evaluation

### Artifact Claims

| ID | Claim | Default scope | Reference | Expected runtime |
| --- | --- | --- | --- | ---: |
| C1 | Reconstruction privacy with competitive task utility | CIFAR-10, six methods | Figure 4, Figure 9 | 30 s |
| C2 | Reduced private-attribute inference | CelebA, six methods | Figure 5 | 7 min 12 s |
| C3 | Resistance to likelihood-maximization reconstruction | CIFAR-10, 16 images | Figure 6 | 9 min 52 s |
| C4 | Complementary contribution of MARIO's three stages | CelebA, five variants | Table 3, Figure 10 | 2 min 46 s |
| C5 | `z_priv` is more reconstructive than `z_pub` | CIFAR-10, matched probes | Figure 11 | 3 min 1 s |

The runtime estimates above are based on measured wall-clock time using each default `run.sh` on one
NVIDIA RTX PRO 6000 Blackwell Max-Q GPU. The five experiments took approximately
23 min 22 s in total. Runtime may vary with hardware, software versions, and
dataset storage performance.

The full claim-to-output mapping is in [`docs/CLAIMS.md`](docs/CLAIMS.md).

### Experiment 1: Privacy and Utility

Evaluate test accuracy and the supervised reconstruction attack over all 10,000
CIFAR-10 test images. The script reports PSNR, SSIM, and AlexNet LPIPS and saves
qualitative reconstruction grids.

```bash
./experiments/01_main_privacy_utility/run.sh
```

### Experiment 2: Property Inference

Evaluate Male and Young inference on all 39,829 CelebA evaluation images:

```bash
./experiments/02_property_inference/run.sh
```

### Experiment 3: Likelihood Maximization

Run the white-box input-optimization attack used in the paper:

```bash
./experiments/03_likelihood_maximization/run.sh
```

### Experiment 4: Stage-Wise Ablation

Evaluate Vanilla, channel obfuscation only, latent decomposition only,
variational sampling only, and full MARIO on CelebA:

```bash
./experiments/04_ablation/run.sh
```

### Experiment 5: Latent Decomposition

Train matched reconstruction probes on frozen `z_priv` and `z_pub`:

```bash
./experiments/05_latent_decomposition/run.sh
```

### Optional: Retrain Evaluation Attackers

Released attacker checkpoints are used by default. Reconstruction and property attackers can optionally be retrained while all victim models remain frozen. See [`docs/TRAINING.md`](docs/TRAINING.md) for commands, settings, and output paths.

### Outputs

Generated files are written under `generated_results/<experiment>/`. Reference
CSV files are under `expected_results/`. Stochastic defenses and optimized probes
may vary slightly across GPU and library versions, so validation should compare
rounded values and the reported direction of each result rather than require
bitwise-identical floats.

<br>

## Artifact Usage

### Notebook and Core Scripts

[`MARIO_CIFAR10_artifact.ipynb`](MARIO_CIFAR10_artifact.ipynb) demonstrates
single-image CIFAR-10 prediction and reconstruction from the server-exposed
`x_pub`. Equivalent command-line entry points are:

```bash
python scripts/infer.py --index 0
python scripts/reconstruct.py --num-images 10
```

### Optional: Training from Scratch

Core CIFAR-10 training remains available independently of the checkpoint-based artifact experiments:

```bash
python scripts/train.py
python scripts/train_reconstruction.py --checkpoint runs/cifar10_mario/best.pt --save-images 10
```

See [`docs/TRAINING.md`](docs/TRAINING.md) for default hyperparameters, checkpoint selection, optional evaluation-attacker training, and generated files.

### Additional Documentation

Implementation-specific metric, checkpoint, baseline, and security notes are in [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md).

The authors' public-release commitment is recorded in
[`docs/PUBLIC_RELEASE.md`](docs/PUBLIC_RELEASE.md).
