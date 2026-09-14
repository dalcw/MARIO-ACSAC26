import argparse
import sys
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mario.checkpoint import load_mario_checkpoint, load_reconstruction_checkpoint
from mario.data import get_cifar10_loaders


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_reconstruction_images(args):
    device = resolve_device(args.device)
    _, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )

    model, _ = load_mario_checkpoint(args.checkpoint, device=device)
    attacker, _ = load_reconstruction_checkpoint(args.recon_checkpoint, device=device)
    model.eval()
    attacker.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    originals = []
    reconstructions = []

    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            out = model(X)
            X_hat = attacker(out["x_pub"]).clamp(0, 1)

            n = min(args.num_images - saved, X.size(0))
            for i in range(n):
                idx = saved + i
                original = X[i].detach().cpu().clamp(0, 1)
                reconstruct = X_hat[i].detach().cpu().clamp(0, 1)
                save_image(original, output_dir / f"{idx}_original.png")
                save_image(reconstruct, output_dir / f"{idx}_reconstruct.png")
                originals.append(original)
                reconstructions.append(reconstruct)
            saved += n
            if saved >= args.num_images:
                break

    if originals:
        grid = make_grid(torch.stack(originals + reconstructions, dim=0), nrow=len(originals), padding=2)
        save_image(grid, output_dir / "grid_original_then_reconstruct.png")

    print(f"Saved {saved} reconstruction pairs to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run reconstruction inference from MARIO x_pub.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "release" / "cifar10" / "defenses" / "mario.pt"))
    parser.add_argument("--recon-checkpoint", default=str(ROOT / "checkpoints" / "release" / "cifar10" / "reconstruction_attackers" / "our_attacker.pt"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "generated_results" / "demo_reconstruction"))
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    save_reconstruction_images(parse_args())
