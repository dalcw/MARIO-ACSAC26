import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mario.checkpoint import save_mario_checkpoint
from mario.data import get_cifar10_loaders
from mario.model import build_mario_cifar10
from mario.utils import kl_normal


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, test_loader, device):
    model.eval()
    corrects = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = out["preds"].argmax(dim=1)
            corrects += (preds == y).sum().item()
            total += y.size(0)

    return corrects / total


def train(args):
    device = resolve_device(args.device)
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )

    model = build_mario_cifar10(
        num_classes=10,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best_acc = -1.0

    for epoch in range(args.epochs):
        model.train()

        losses = 0.0
        task_losses = 0.0
        priv_losses = 0.0
        pub_losses = 0.0

        for X, y in tqdm.tqdm(train_loader, desc=f"EPOCH - {epoch + 1}"):
            X, y = X.to(device), y.to(device)

            out = model(X)
            preds = out["preds"]
            smashed_data = out["smashed_data"]
            x_priv_rec = out["x_priv_rec"]
            z_mu = out["z_mu"]
            z_logvar = out["z_logvar"]

            optimizer.zero_grad()

            L_task = F.cross_entropy(preds, y)
            L_priv_rec = F.mse_loss(x_priv_rec, smashed_data)
            L_pub_kl = kl_normal(z_mu, z_logvar)
            loss = L_task + args.alpha * L_priv_rec + args.beta * L_pub_kl

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            losses += loss.item()
            task_losses += L_task.item()
            priv_losses += L_priv_rec.item()
            pub_losses += L_pub_kl.item()

        n_batches = len(train_loader)
        test_acc = evaluate(model, test_loader, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": losses / n_batches,
            "task_loss": task_losses / n_batches,
            "priv_recon_loss": priv_losses / n_batches,
            "pub_kl_loss": pub_losses / n_batches,
            "test_accuracy": test_acc,
        }
        rows.append(row)
        write_log(out_dir / "train_log.csv", rows)

        print(f"Training Loss     : {row['train_loss']:.4f}")
        print(f"  Task Loss       : {row['task_loss']:.4f}")
        print(f"  Priv recon Loss : {row['priv_recon_loss']:.4f}")
        print(f"  Pub KL Loss     : {row['pub_kl_loss']:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        save_mario_checkpoint(out_dir / "last.pt", model, optimizer, epoch + 1, row, args)
        if test_acc > best_acc:
            best_acc = test_acc
            save_mario_checkpoint(out_dir / "best.pt", model, optimizer, epoch + 1, row, args)

    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train MARIO on CIFAR-10.")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "cifar10_mario"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
