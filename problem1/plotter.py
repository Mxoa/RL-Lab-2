import argparse
import re
from pathlib import Path
import numpy as np
import torch

#!/usr/bin/env python3
# plotter.py
# Charge un modèle PyTorch (fichier 'neural-network-1.pth') puis trace une surface 3D
# de l'action en fonction de la hauteur y et de l'angle theta.
# Usage: python plotter.py --path neural-network-1.pth


import matplotlib.pyplot as plt
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- required for 3D projection


def try_load_model(path: str):
    """
    Essaie de charger le fichier .pth. Si le fichier contient un objet nn.Module
    il est renvoyé. Si c'est un state_dict, on reconstruit un MLP simple
    en inférant les tailles depuis les poids.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, nn.Module):
        return obj

    # si c'est un dictionnaire de checkpoint
    if isinstance(obj, dict):
        # certains checkpoints encapsulent le state_dict sous la clé 'state_dict'
        if "state_dict" in obj:
            sd = obj["state_dict"]
        else:
            sd = obj

        # si sd contient des clés comme 'module.' découper le préfixe commun
        keys = list(sd.keys())
        if not keys:
            raise RuntimeError("state_dict vide")

        # enlever préfixe common (ex: 'module.' ou 'net.')
        prefix = None
        m = re.match(r"^(.*?)(\d+\.weight)$", keys[0])
        if m:
            # pas de préfixe si déjà direct
            prefix = ""
        else:
            # trouver le plus petit préfix qui rend tous les keys similaires
            # heuristique: si keys start with same token like 'module.'
            common = None
            for k in keys:
                parts = k.split(".")
                if common is None:
                    common = parts[0]
                elif common != parts[0]:
                    common = ""
                    break
            prefix = (common + ".") if common else ""

        # récupérer toutes les weight keys triées d'apparition
        weight_keys = [k for k in keys if k.endswith(".weight")]
        weight_keys.sort()
        sizes = []
        for k in weight_keys:
            w = sd[k]
            if w.ndim != 2:
                # ignorer conv ou autres ; on ne gère que couches lineaires
                continue
            sizes.append((w.shape[1], w.shape[0]))  # (in, out) -> we'll transpose to (in,out)

        if not sizes:
            raise RuntimeError("Aucune couche linéaire détectée dans le state_dict.")

        # construire un MLP basé sur sizes déduites
        layers = []
        for i, (in_dim, out_dim) in enumerate(sizes):
            layers.append(nn.Linear(in_dim, out_dim))
            # ajouter activation si pas dernière couche
            if i < len(sizes) - 1:
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        # tenter de charger le state_dict
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            # mapping heuristique : construire nouveau dict compatible
            new_sd = {}
            model_keys = list(model.state_dict().keys())
            sd_items = [sd[k] for k in weight_keys] + []
            # fallback: charger par correspondance d'ordre pour les weights et biases
            sd_values = []
            for k in keys:
                if k.endswith(".weight") or k.endswith(".bias"):
                    sd_values.append(sd[k])
            # assigner dans l'ordre
            for k, v in zip(model_keys, sd_values):
                new_sd[k] = v
            model.load_state_dict(new_sd, strict=False)
        return model

    raise RuntimeError("Format de fichier non supporté pour le chargement du modèle.")


def evaluate_grid(model, y_vals, theta_vals, device="cpu"):
    """
    Évalue le modèle sur une grille (y, theta).
    On suppose que l'entrée du modèle est [y, theta] dans cet ordre.
    Si le modèle renvoie plusieurs valeurs, on prend la première composante.
    """
    model.to(device)
    model.eval()
    Y, T = np.meshgrid(y_vals, theta_vals, indexing="xy")
    pts = np.stack([Y.ravel(), T.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        inp = torch.from_numpy(pts).to(device)
        try:
            out = model(inp)
        except Exception as e:
            # tenter avec transpose si modèle attend (batch, features) mais features inversées
            inp2 = inp[:, [1, 0]]
            out = model(inp2)
        if isinstance(out, tuple) or isinstance(out, list):
            out = out[0]
        out = out.cpu().numpy()
    # si sortie multi-dim, prendre première composante
    if out.ndim == 2 and out.shape[1] > 1:
        out = out[:, 0]
    Z = out.reshape(Y.shape)
    return Y, T, Z


def plot_surface(Y, T, Z, title="Action en fonction de y et theta", save_to=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Y, T, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("y (hauteur)")
    ax.set_ylabel("theta (angle, rad)")
    ax.set_zlabel("action (sortie du modèle)")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.6, aspect=12)
    if save_to:
        fig.savefig(save_to, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Trace l'action d'un réseau sur une grille (y, theta).")
    parser.add_argument("--path", type=str, default="neural-network-1.pth", help="chemin vers le .pth")
    parser.add_argument("--ymin", type=float, default=-1.0)
    parser.add_argument("--ymax", type=float, default=1.0)
    parser.add_argument("--ny", type=int, default=121)
    parser.add_argument("--thetamin", type=float, default=-3.14159)
    parser.add_argument("--thetamax", type=float, default=3.14159)
    parser.add_argument("--ntheta", type=int, default=121)
    parser.add_argument("--save", type=str, default=None, help="fichier de sortie PNG (optionnel)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {args.path}")

    model = try_load_model(str(path))

    y_vals = np.linspace(args.ymin, args.ymax, args.ny)
    theta_vals = np.linspace(args.thetamin, args.thetamax, args.ntheta)

    Y, T, Z = evaluate_grid(model, y_vals, theta_vals, device="cpu")
    plot_surface(Y, T, Z, save_to=args.save)


if __name__ == "__main__":
    main()