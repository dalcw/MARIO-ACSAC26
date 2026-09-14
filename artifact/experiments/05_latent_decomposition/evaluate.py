#!/usr/bin/env python3
"""Train matched reconstruction probes on MARIO z_pub and z_priv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from mario.artifact_data import build_dataset
from mario.artifact_metrics import ReconstructionMetrics
from mario.defenses import ClientResNet18
from mario.evaluation import load_torch, seed_everything, select_device, write_csv
from mario.evaluation import print_expected_comparison
from mario.latent_decomposition import DeepConvImageDecoder, MarioLatentDecomposition


REPRESENTATIONS = ("z_priv", "z_pub")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints" / "release")
    parser.add_argument("--output", type=Path, default=ROOT / "generated_results" / "05_latent_decomposition")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-test-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--save-images", type=int, default=8)
    parser.add_argument("--reuse-decoders", action="store_true")
    parser.add_argument("--download-cifar10", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def cache_latents(client, decomposition, dataset, batch_size, workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    collected = {name: [] for name in REPRESENTATIONS}
    images = []
    for batch in tqdm(loader, desc="cache frozen latents"):
        batch_images = batch[0].to(device, non_blocking=True)
        output = decomposition.latents(client(batch_images))
        for name in REPRESENTATIONS:
            collected[name].append(output[name].cpu())
        images.append(batch_images.cpu())
    return {name: torch.cat(parts) for name, parts in collected.items()}, torch.cat(images)


def train_decoder(args, representation, latent, images, device):
    decoder = DeepConvImageDecoder(z_dim=latent.shape[1], image_size=32).to(device)
    checkpoint = args.output / "decoders" / f"{representation}.pt"
    if args.reuse_decoders:
        decoder.load_state_dict(load_torch(checkpoint, device))
        return decoder

    dataset = TensorDataset(latent, images)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(loader, desc=f"train {representation} decoder {epoch}/{args.epochs}")
        for batch_latent, batch_images in progress:
            batch_latent = batch_latent.to(device, non_blocking=True)
            batch_images = batch_images.to(device, non_blocking=True)
            reconstruction = decoder(batch_latent)
            loss = F.mse_loss(reconstruction, batch_images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch_images.shape[0]
            sample_count += batch_images.shape[0]
            progress.set_postfix(mse=f"{loss_sum / sample_count:.6f}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), checkpoint)
    return decoder


@torch.inference_mode()
def evaluate_decoder(args, representation, decoder, latent, images, device):
    dataset = TensorDataset(latent, images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    metrics = ReconstructionMetrics(device, lpips_net="alex")
    saved_images = []
    saved_reconstructions = []
    decoder.eval()
    for batch_latent, batch_images in tqdm(loader, desc=f"evaluate {representation}"):
        batch_latent = batch_latent.to(device, non_blocking=True)
        batch_images = batch_images.to(device, non_blocking=True)
        reconstruction = decoder(batch_latent).clamp(0, 1)
        metrics.update(reconstruction, batch_images)
        remaining = args.save_images - sum(value.shape[0] for value in saved_images)
        if remaining > 0:
            saved_images.append(batch_images[:remaining].cpu())
            saved_reconstructions.append(reconstruction[:remaining].cpu())

    originals = torch.cat(saved_images)
    reconstructions = torch.cat(saved_reconstructions)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_image(
        make_grid(torch.cat([originals, reconstructions]), nrow=originals.shape[0], padding=1),
        figure_dir / f"{representation}_original_then_reconstruction.png",
    )
    return {
        "representation": representation,
        "test_samples": metrics.count,
        "decoder_epochs": args.epochs,
        **metrics.compute(),
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    seed_everything(args.seed)
    train_dataset = build_dataset("cifar10", args.data_root, "train", args.download_cifar10)
    test_dataset = build_dataset("cifar10", args.data_root, "test", args.download_cifar10)
    train_dataset = Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
    test_dataset = Subset(test_dataset, range(min(args.max_test_samples, len(test_dataset))))

    client = ClientResNet18(input_size=32).to(device)
    mario_checkpoint = load_torch(
        args.checkpoint_root / "cifar10" / "defenses" / "mario.pt", device
    )
    client.load_state_dict(mario_checkpoint["client_state_dict"])
    client.eval()

    decomposition = MarioLatentDecomposition().to(device)
    decomposition_checkpoint = load_torch(
        args.checkpoint_root / "latent_decomposition" / "cifar10" / "decomposition.pt",
        device,
    )
    decomposition.load_state_dict(decomposition_checkpoint["model_state_dict"])
    decomposition.eval()
    for module in (client, decomposition):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    train_latents, train_images = cache_latents(
        client, decomposition, train_dataset, args.batch_size, args.workers, device
    )
    test_latents, test_images = cache_latents(
        client, decomposition, test_dataset, args.batch_size, args.workers, device
    )
    del client, decomposition, mario_checkpoint, decomposition_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for representation in REPRESENTATIONS:
        seed_everything(args.seed)
        decoder = train_decoder(
            args, representation, train_latents[representation], train_images, device
        )
        row = evaluate_decoder(
            args, representation, decoder, test_latents[representation], test_images, device
        )
        rows.append(row)
        print(row, flush=True)
        write_csv(args.output / "metrics.csv", rows)
        del decoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_expected_comparison(
        rows,
        REPOSITORY_ROOT / "claims" / "claim5" / "expected" / "metrics.csv",
        key_fields=("representation",),
        metric_fields=("psnr", "ssim", "lpips"),
    )
    print(f"\033[94mResults saved to: {args.output.resolve()}\033[0m", flush=True)


if __name__ == "__main__":
    main()
