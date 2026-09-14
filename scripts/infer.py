import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mario.checkpoint import load_mario_checkpoint

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def resolve_device(device):
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def predict_image(model, image_path, device):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)["preds"]
        prob = torch.softmax(logits, dim=1)[0]
        pred = int(prob.argmax().item())

    print(f"Prediction: {pred} ({CIFAR10_CLASSES[pred]})")
    print(f"Confidence: {float(prob[pred]):.4f}")


def predict_cifar_index(model, data_dir, index, device, download):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=download)
    x, y = dataset[index]
    x = x.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)["preds"]
        prob = torch.softmax(logits, dim=1)[0]
        pred = int(prob.argmax().item())

    print(f"Index: {index}")
    print(f"Target: {int(y)} ({CIFAR10_CLASSES[int(y)]})")
    print(f"Prediction: {pred} ({CIFAR10_CLASSES[pred]})")
    print(f"Confidence: {float(prob[pred]):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run MARIO inference on CIFAR-10.")
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "release" / "cifar10" / "defenses" / "mario.pt"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--image", default=None)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model, _ = load_mario_checkpoint(args.checkpoint, device=device)

    download = not args.no_download
    if args.image is not None:
        predict_image(model, args.image, device)
    else:
        predict_cifar_index(model, args.data_dir, args.index, device, download)


if __name__ == "__main__":
    main()
