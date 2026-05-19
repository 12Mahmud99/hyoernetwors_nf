import architectures.cnn as cnn
import torch
import argparse
import json
from architectures.utils import load_cifar_10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN on CIFAR-10')
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--output", "-o", type=str, help="Path to save the best model (overrides config)")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    device = args.device if torch.cuda.is_available() else 'cpu'

    trainloader, valloader, testloader = load_cifar_10(batch_size=config.get("batch_size", 64))

    model = cnn.ConvNet()

    out_path = args.output if args.output else config.get("out_path", "best_model.pth")

    model.fit(
        train_loader=trainloader,
        val_loader=valloader,
        epochs=config.get("epochs", 10),
        lr=config.get("lr", 0.001),
        device=device,
        val_every=config.get("val_every", 5),
        patience=config.get("patience", 5),
        out_path=out_path
    )

    print(f"Training completed. Best model saved to {out_path}")