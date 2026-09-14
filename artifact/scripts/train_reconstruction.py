import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torchvision.utils import make_grid, save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mario.checkpoint import load_mario_checkpoint, save_reconstruction_checkpoint
from mario.data import get_cifar10_loaders
from mario.reconstruction import SmashedReconstructionAttacker


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    device = resolve_device(args.device)
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )

    model, _ = load_mario_checkpoint(args.checkpoint, device=device)
    model.eval()

    attacker = SmashedReconstructionAttacker(in_channels=128, img_channels=3).to(device)
    optimizer = optim.Adam(attacker.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    last_loss = 0.0
    for epoch in range(args.epochs):
        attacker.train()
        total_loss = 0.0

        for X, _ in tqdm.tqdm(train_loader, desc=f"[Attacker Train] Epoch {epoch + 1}"):
            X = X.to(device)

            with torch.no_grad():
                out = model(X)
                x_pub = out["x_pub"]

            x_hat = attacker(x_pub)
            loss = F.mse_loss(x_hat, X)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(attacker.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        last_loss = total_loss / len(train_loader)
        print(f"[Attacker] Epoch {epoch + 1} | Recon Loss: {last_loss:.4f}")
        save_reconstruction_checkpoint(out_dir / "reconstruction_attacker_last.pt", attacker, optimizer, epoch + 1, last_loss, args)

    if args.save_images > 0:
        save_reconstruction_images(model, attacker, test_loader, device, out_dir / "reconstruction_images", args.save_images)

    save_reconstruction_checkpoint(out_dir / "reconstruction_attacker.pt", attacker, optimizer, args.epochs, last_loss, args)


def save_reconstruction_images(model, attacker, test_loader, device, image_dir, num_images):
    image_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    attacker.eval()

    X, _ = next(iter(test_loader))
    X = X.to(device)[:num_images]
    with torch.no_grad():
        x_pub = model(X)["x_pub"]
        X_hat = attacker(x_pub)

    X_orig_vis = X.clamp(0, 1).cpu()
    X_hat_vis = X_hat.clamp(0, 1).cpu()

    for i in range(X_orig_vis.size(0)):
        save_image(X_orig_vis[i], image_dir / f"{i}_original.png")
        save_image(X_hat_vis[i], image_dir / f"{i}_reconstruct.png")

    grid = make_grid(torch.cat([X_orig_vis, X_hat_vis], dim=0), nrow=X_orig_vis.size(0), padding=2)
    save_image(grid, image_dir / "grid_original_then_reconstruct.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reconstruction attacker from MARIO x_pub.")
    parser.add_argument("--checkpoint", default=str(ROOT / "runs" / "cifar10_mario" / "pretrained.pt"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "cifar10_reconstruction"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--save-images", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
