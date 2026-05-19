import architectures.cnn as cnn    
import torch 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained CNN on CIFAR-10')
    parser.add_argument("--config", "-c", type=int, default=10)
    parser.add_argument("--ckpt", "--checkpoint", "--model", types=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--device", "-d", type=str, default="cuda")
    
    device = args.device
    model_path=args.ckpt
    
    
    model = cnn.ConvNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    