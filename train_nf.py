# train_nf.py
import os
import torch
import torch.utils.data as data
import zuko
import numpy as np
import argparse
from glob import glob

class TrajectoryDataset(data.Dataset):
    def __init__(self, file_list, norm_stats=None):
        self.file_list = file_list
        self.norm_stats = norm_stats  

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        theta_t = data['theta_t'].astype(np.float32)
        grad_t  = data['grad_t'].astype(np.float32)
        theta_next = data['theta_next'].astype(np.float32)

        if self.norm_stats is not None:
            theta_t = (theta_t - self.norm_stats['theta_mean']) / self.norm_stats['theta_std']
            grad_t  = (grad_t  - self.norm_stats['grad_mean'])  / self.norm_stats['grad_std']
            theta_next = (theta_next - self.norm_stats['next_mean']) / self.norm_stats['next_std']

        ctx = np.stack([theta_t, grad_t], axis=1)  
        tgt = theta_next[:, None]                

        return torch.from_numpy(ctx), torch.from_numpy(tgt)

def collate_fn(batch):
    ctx_list, tgt_list = zip(*batch)
    ctx = torch.cat(ctx_list, dim=0)   
    tgt = torch.cat(tgt_list, dim=0)   
    return ctx, tgt

def compute_stats(file_list):
    all_theta = []
    all_grad  = []
    all_next  = []
    for f in file_list:
        data = np.load(f)
        all_theta.append(data['theta_t'])
        all_grad.append(data['grad_t'])
        all_next.append(data['theta_next'])
    all_theta = np.concatenate(all_theta)
    all_grad  = np.concatenate(all_grad)
    all_next  = np.concatenate(all_next)
    stats = {
        'theta_mean': all_theta.mean(),
        'theta_std':  all_theta.std(),
        'grad_mean':  all_grad.mean(),
        'grad_std':   all_grad.std(),
        'next_mean':  all_next.mean(),
        'next_std':   all_next.std(),
    }
    for key in ['theta_std', 'grad_std', 'next_std']:
        if stats[key] < 1e-8:
            stats[key] = 1.0
    return stats

def train_nf(train_loader, val_loader, device, epochs, lr, transforms, hidden, out_path):
    flow = zuko.flows.NSF(features=1, transforms=transforms, context=2, hidden_features=hidden)
    flow.to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        flow.train()
        total_loss = 0.0
        n_samples = 0
        for ctx, tgt in train_loader:
            ctx, tgt = ctx.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss = -flow(ctx).log_prob(tgt).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ctx.size(0)
            n_samples += ctx.size(0)
        train_loss = total_loss / n_samples

        if (epoch + 1) % 5 == 0:
            flow.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for ctx, tgt in val_loader:
                    ctx, tgt = ctx.to(device), tgt.to(device)
                    loss = -flow(ctx).log_prob(tgt).mean()
                    val_loss += loss.item() * ctx.size(0)
                    n_val += ctx.size(0)
            val_loss /= n_val
            print(f"Epoch {epoch+1:3d} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(flow.state_dict(), out_path)
        else:
            print(f"Epoch {epoch+1:3d} | Train loss: {train_loss:.6f}")

    print(f"Training finished. Best model saved to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', type=str, required=True, help='Directory containing .npz trajectory files')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of files for validation')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of transitions per batch')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--transforms', type=int, default=5)
    parser.add_argument('--hidden', type=int, nargs=2, default=[64,64])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='nf_optimizer.pt')
    args = parser.parse_args()

    file_list = glob(os.path.join(args.traj_dir, '*.npz'))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No .npz files found in {args.traj_dir}")
    print(f"Found {len(file_list)} trajectory files")

    np.random.seed(42)
    indices = np.random.permutation(len(file_list))
    split = int(len(file_list) * (1 - args.val_split))
    train_files = [file_list[i] for i in indices[:split]]
    val_files   = [file_list[i] for i in indices[split:]]
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    stats = compute_stats(train_files)
    print("Normalization statistics computed.")

    train_dataset = TrajectoryDataset(train_files, norm_stats=stats)
    val_dataset   = TrajectoryDataset(val_files,   norm_stats=stats)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    stats_path = os.path.splitext(args.output)[0] + '_stats.npz'
    np.savez(stats_path, **stats)
    print(f"Saved normalization statistics to {stats_path}")

    train_nf(train_loader, val_loader, device=args.device, epochs=args.epochs, lr=args.lr,
             transforms=args.transforms, hidden=tuple(args.hidden), out_path=args.output)

if __name__ == '__main__':
    main()