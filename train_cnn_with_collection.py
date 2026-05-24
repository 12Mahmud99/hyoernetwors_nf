import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
from architectures.cnn import ConvNet 

def flatten_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

def flatten_grads(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)

def get_fixed_train_loader(batch_size=128, val_split=0.1, seed=42):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    torch.manual_seed(seed)
    train_size = int((1 - val_split) * len(full_trainset))
    _, _ = random_split(full_trainset, [train_size, len(full_trainset) - train_size])  # we only need training subset
    # Actually we want the training subset; let's do:
    train_subset, _ = random_split(full_trainset, [train_size, len(full_trainset) - train_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader

def collect_trajectories(num_runs=100, steps_per_run=1000, batch_size=128, lr=0.001,
                         device='cuda', save_dir='trajectories'):
    os.makedirs(save_dir, exist_ok=True)
    train_loader = get_fixed_train_loader(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    batch_list = list(train_loader) 
    num_batches = len(batch_list)
    print(f"Number of batches available: {num_batches}")

    for run_id in range(num_runs):
        torch.manual_seed(run_id + 42)
        torch.cuda.manual_seed_all(run_id + 42)

        model = ConvNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for step in range(steps_per_run):
            images, labels = batch_list[step % num_batches]
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            theta_t = flatten_params(model)
            grad_t = flatten_grads(model)

            optimizer.step()
            theta_next = flatten_params(model)

            np.savez(os.path.join(save_dir, f'run{run_id:04d}_step{step:06d}.npz'),
                     theta_t=theta_t.detach().cpu().numpy(),
                     grad_t=grad_t.detach().cpu().numpy(),
                     theta_next=theta_next.detach().cpu().numpy())

            if step % 500 == 0:
                print(f"Run {run_id}, step {step}: saved")
        print(f"Run {run_id+1}/{num_runs} completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--steps_per_run', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='trajectories')
    args = parser.parse_args()
    collect_trajectories(**vars(args))