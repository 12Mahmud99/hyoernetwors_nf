# forecast.py
import torch
import zuko
import numpy as np
import argparse

def load_model(model_path, prediction_length, context_length, device='cpu'):
    flow = zuko.flows.NSF(
        features=prediction_length,
        transforms=3,
        context=context_length,
        hidden_features=(64, 64)
    )
    state_dict = torch.load(model_path, map_location=device)
    flow.load_state_dict(state_dict)
    flow.to(device)
    flow.eval()
    return flow

def forecast_on_test_data(
    model, data, context_length, prediction_length,
    num_windows=None, stride=1, num_samples=1, device='cpu'
):
    """
    data: torch.Tensor of shape (N, T) – already normalized
    """
    total_steps = data.shape[1]

    start_indices = []
    for start in range(0, total_steps - context_length - prediction_length + 1, stride):
        start_indices.append(start)
    if num_windows is not None:
        start_indices = start_indices[:num_windows]

    results = []
    with torch.no_grad():
        for start in start_indices:
            # Context (normalized)
            context = data[:, start:start+context_length].to(device)
            cond_dist = model(context)
            samples = cond_dist.sample(torch.Size([num_samples]))  # (num_samples, N, pred_len)

            # Ground truth full trajectory for this window: context + future
            future = data[:, start+context_length:start+context_length+prediction_length]
            full_trajectory = torch.cat([context.cpu(), future], dim=1)  # (N, context_len+pred_len)

            results.append({
                'start_idx': start,
                'full_ground_truth': full_trajectory.numpy(),   # includes both context and future
                'forecasts': samples.cpu().numpy()
            })
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--test_data", "-d", required=True,
                        help="Path to test .npz file containing 'positions' (already normalized)")
    parser.add_argument("--context_length", "-c", type=int, required=True)
    parser.add_argument("--prediction_length", "-p", type=int, required=True)
    parser.add_argument("--num_windows", "-n", type=int, default=None)
    parser.add_argument("--stride", "-s", type=int, default=1)
    parser.add_argument("--num_samples", "-k", type=int, default=10)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", "-o", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device

    data_np = np.load(args.test_data)
    if "positions" not in data_np:
        raise KeyError("Test .npz must contain 'positions' array (already normalized)")
    data = torch.from_numpy(data_np["positions"]).float()   # (N, T)

    model = load_model(args.model, args.prediction_length, args.context_length, device)

    results = forecast_on_test_data(
        model, data,
        args.context_length, args.prediction_length,
        args.num_windows, args.stride, args.num_samples, device
    )

    print(f"Generated {len(results)} forecasts.")
    for i, res in enumerate(results):
        print(f"Window {i}: start_idx={res['start_idx']}, "
              f"forecast shape={res['forecasts'].shape}, "
              f"full_ground_truth shape={res['full_ground_truth'].shape}")

    if args.output:
        start_indices = np.array([r['start_idx'] for r in results])
        full_ground_truths = np.stack([r['full_ground_truth'] for r in results])
        forecasts = np.stack([r['forecasts'] for r in results])
        np.savez(args.output,
                 start_indices=start_indices,
                 full_ground_truth=full_ground_truths,
                 forecasts=forecasts)
        print(f"Saved forecasts and combined ground truth to {args.output}")

if __name__ == "__main__":
    main()