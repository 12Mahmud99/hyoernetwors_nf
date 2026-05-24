import architectures.cnn as cnn    
import torch 
import argparse
from architectures.utils import load_cifar_10
import json

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Evaluate a trained CNN on CIFAR-10')
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--ckpt", "--checkpoint", "--model", "-m", type=str, required=True) 
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--device", "-d", type=str, default="cuda")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        configs = json.load(f)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    model_path = args.ckpt
    
    _, _, test_loader = load_cifar_10(configs["batch_size"])
    
    model = cnn.ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    test_loss, test_acc = model.evaluate(test_loader, device=device)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.2f}%\n")