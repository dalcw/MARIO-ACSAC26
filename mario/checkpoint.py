import torch

from .model import build_mario_cifar10
from .reconstruction import SmashedReconstructionAttacker


def save_mario_checkpoint(path, model, optimizer, epoch, metrics, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "args": vars(args) if hasattr(args, "__dict__") else args,
        },
        path,
    )


def load_mario_checkpoint(path, device="cpu", num_classes=10, embed_dim=128, dropout=0.1):
    payload = torch.load(path, map_location=device, weights_only=True)
    saved_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    embed_dim = int(saved_args.get("embed_dim", embed_dim))
    dropout = float(saved_args.get("dropout", dropout))
    model = build_mario_cifar10(num_classes=num_classes, embed_dim=embed_dim, dropout=dropout).to(device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    elif isinstance(payload, dict) and {"client_state_dict", "adapter_state_dict", "server_state_dict"}.issubset(payload):
        model.client_model.load_state_dict(payload["client_state_dict"])
        model.adapter_model.load_state_dict(payload["adapter_state_dict"])
        model.server_model.load_state_dict(payload["server_state_dict"])
    elif isinstance(payload, dict):
        model.load_state_dict(payload)
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(payload)}")
    return model, payload


def save_reconstruction_checkpoint(path, attacker, optimizer, epoch, loss, args):
    torch.save(
        {
            "epoch": epoch,
            "attacker_state_dict": attacker.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "loss": loss,
            "args": vars(args) if hasattr(args, "__dict__") else args,
        },
        path,
    )


def load_reconstruction_checkpoint(path, device="cpu", in_channels=128, img_channels=3):
    payload = torch.load(path, map_location=device, weights_only=True)
    attacker = SmashedReconstructionAttacker(in_channels=in_channels, img_channels=img_channels).to(device)

    if isinstance(payload, dict) and "attacker_state_dict" in payload:
        attacker.load_state_dict(payload["attacker_state_dict"])
    elif isinstance(payload, dict):
        attacker.load_state_dict(payload)
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(payload)}")
    return attacker, payload
