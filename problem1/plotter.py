#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def try_load_model(path: str):
    """Loads the PyTorch model and handles state_dict or nn.Module."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj

    if isinstance(obj, dict):
        sd = obj["state_dict"] if "state_dict" in obj else obj
        keys = list(sd.keys())
        
        # Infer simple MLP structure if necessary
        weight_keys = [k for k in keys if k.endswith(".weight")]
        weight_keys.sort()
        sizes = []
        for k in weight_keys:
            w = sd[k]
            if w.ndim == 2:
                sizes.append((w.shape[1], w.shape[0]))

        layers = []
        for i, (in_dim, out_dim) in enumerate(sizes):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(sizes) - 1:
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        model.load_state_dict(sd, strict=False)
        return model
    raise RuntimeError("Unsupported file format.")

def evaluate_grid(model, y_vals, omega_vals, mode="value", device="cpu"):
    """
    Evaluates the model on a grid (y, omega).
    The state vector sent is [0, y, 0, 0, omega, 0, 0, 0].
    """
    model.to(device)
    model.eval()
    Y, O = np.meshgrid(y_vals, omega_vals, indexing="xy")
    
    # Prepare 8-D state vector for Lunar Lander
    num_pts = Y.size
    pts_8d = np.zeros((num_pts, 8), dtype=np.float32)
    pts_8d[:, 1] = Y.ravel()     # Height y
    pts_8d[:, 4] = O.ravel()     # Angle omega (rad)
    
    with torch.no_grad():
        inp = torch.from_numpy(pts_8d).to(device)
        q_values = model(inp)
        
        if mode == "value":
            # Plot (1): Maximum state value
            res = torch.max(q_values, dim=1)[0]
        else:
            # Plot (2): Optimal action (index 0 to 3)
            res = torch.argmax(q_values, dim=1)
            
        out = res.cpu().numpy()
    
    Z = out.reshape(Y.shape)
    return Y, O, Z

def plot_surface(Y, O, Z, title, z_label, save_to=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    cmap = "viridis"
    
    surf = ax.plot_surface(Y, O, Z, cmap=cmap, edgecolor='none', alpha=0.9)
    
    ax.set_xlabel("Y (Height)")
    ax.set_ylabel("Omega (Angle, radians)")
    ax.set_zlabel(z_label)
    ax.set_title(title)
    
    if "Value" in title:
        fig.colorbar(surf, shrink=0.6, aspect=12)
    
    # Adjust view to match captures
    ax.view_init(elev=20, azim=-60)
    
    if save_to:
        fig.savefig(save_to, dpi=200, bbox_inches="tight")
        print(f"Plot saved to: {save_to}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="neural-network-1.pth")
    parser.add_argument("--ymin", type=float, default=0.0)      # According to problem statement
    parser.add_argument("--ymax", type=float, default=1.5)      # According to problem statement
    parser.add_argument("--thetamin", type=float, default=-np.pi)
    parser.add_argument("--thetamax", type=float, default=np.pi)
    args = parser.parse_args()

    model = try_load_model(args.path)

    y_vals = np.linspace(args.ymin, args.ymax, 100)
    omega_vals = np.linspace(args.thetamin, args.thetamax, 100)

    # 1. Generate Value Plot (max Q)
    print("Generating value surface (V-shape)...")
    Y, O, Z_val = evaluate_grid(model, y_vals, omega_vals, mode="value")
    plot_surface(Y, O, Z_val, "DQN Policy Visualization - Max Q-values", "Q-values", save_to="plots/value_plot.png")

    # 2. Generate Action Plot (argmax Q)
    print("Generating action surface (Policy)...")
    Y, O, Z_act = evaluate_grid(model, y_vals, omega_vals, mode="action")
    plot_surface(Y, O, Z_act, "DQN Policy Visualization - Optimal Action", "Action Index", save_to="plots/action_plot.png")

if __name__ == "__main__":
    main()