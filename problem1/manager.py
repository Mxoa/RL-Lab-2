import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import yaml
import numpy as np

# Set LaTeX-friendly style for plots
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'text.usetex': False,  # Set to True if LaTeX is installed
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13
})

# === CONFIGURATION DES EXPÉRIENCES (pour la question e)===
scenarios = [
    # (1) Le Baseline (gamma0)
    {"args": ["--discount-factor", "0.98", "--replay-buffer-size", "20000", "--episodes", "800"], "id": "Baseline"},
    
    # (2) Variations de Gamma
    {"args": ["--discount-factor", "1.0", "--replay-buffer-size", "20000", "--episodes", "800"], "id": "Gamma_1.0"},
    {"args": ["--discount-factor", "0.5", "--replay-buffer-size", "20000", "--episodes", "800"], "id": "Gamma_0.5"},

    # (3) Variations de Mémoire (Replay Buffer)
    {"args": ["--discount-factor", "0.98", "--replay-buffer-size", "500", "--episodes", "800"], "id": "Buffer_500"},
    {"args": ["--discount-factor", "0.98", "--replay-buffer-size", "1000", "--episodes", "800"], "id": "Buffer_1000"},
    {"args": ["--discount-factor", "0.98", "--replay-buffer-size", "20000", "--episodes", "800"], "id": "Buffer_20000"},
    {"args": ["--discount-factor", "0.98", "--replay-buffer-size", "50000", "--episodes", "800"], "id": "Buffer_50000"},
    
    # (3) Variations d'Épisodes
    {"args": ["--discount-factor", "0.98", "--replay-buffer-size", "20000", "--episodes", "200"], "id": "Episodes_200"},
    {"args": ["--discount-factor", "0.98", "--replay-buffer-size", "20000", "--episodes", "400"], "id": "Episodes_400"},
    {"args": ["--discount-factor", "0.98", "--replay-buffer-size", "20000", "--episodes", "800"], "id": "Episodes_800"},
    {"args": ["--discount-factor", "0.98", "--replay-buffer-size", "20000", "--episodes", "1000"], "id": "Episodes_1000"},
]

PYTHON_EXE = sys.executable
SCRIPT_NAME = "DQN_problem.py" 

def get_value(args_list, flag):
    """Récupère la valeur d'un flag dans la liste d'arguments"""
    try:
        return args_list[args_list.index(flag) + 1]
    except (ValueError, IndexError):
        return None

def run_simulation(scenario):
    print(f"\n🚀 Running scenario : {scenario['id']} ...")
    cmd = [PYTHON_EXE, SCRIPT_NAME] + scenario["args"]
    subprocess.run(cmd, check=True)

def get_latest_csv(scenario):
    """Trouve le fichier CSV correspondant aux paramètres du scénario"""
    args = scenario["args"]
    
    # On récupère les valeurs pour reconstruire le nom de fichier
    gamma = float(get_value(args, "--discount-factor"))
    eps = int(get_value(args, "--episodes"))
    mem = int(get_value(args, "--replay-buffer-size"))
    
    expected_path = f"data_experiments/results_gamma{gamma}_mem{mem}_eps{eps}.csv"
    
    if os.path.exists(expected_path):
        return expected_path
    else:
        print(f"⚠️ File not found : {expected_path}")
        return None

def running_average(x, N):
    """Function used to compute the running average of the last N elements of a vector x"""
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def plot_comparison(title, scenario_ids, output_dir="plots", running_avg_window=50):
    """
    Create side-by-side plots for average reward and average steps per episode.
    Format optimized for LaTeX reports (text width, appropriate font size).
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create figure with two subplots side by side (LaTeX text width ~ 6.5-7 inches)
    # Using 12 inches width for two plots side by side (6 inches each, suitable for LaTeX)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))
    
    for s_id in scenario_ids:
        scenario = next((s for s in scenarios if s["id"] == s_id), None)
        if not scenario: 
            continue
        
        csv_path = get_latest_csv(scenario)
        if csv_path:
            df = pd.read_csv(csv_path)
            episodes = df['episode'].values
            
            # Plot average reward (left subplot)
            running_avg_reward = running_average(df['reward'].values, running_avg_window)
            ax[0].plot(episodes, running_avg_reward, label=s_id, linewidth=2)
            
            # Plot average steps (right subplot)
            running_avg_steps = running_average(df['steps'].values, running_avg_window)
            ax[1].plot(episodes, running_avg_steps, label=s_id, linewidth=2)
    
    # Configure left subplot (Reward)
    ax[0].set_xlabel('Episodes', fontsize=11)
    ax[0].set_ylabel('Average Reward', fontsize=11)
    ax[0].set_title('Average Reward vs Episodes', fontsize=12)
    ax[0].legend(fontsize=10)
    ax[0].grid(True, alpha=0.3)
    
    # Configure right subplot (Steps)
    ax[1].set_xlabel('Episodes', fontsize=11)
    ax[1].set_ylabel('Average Number of Steps', fontsize=11)
    ax[1].set_title('Average Steps vs Episodes', fontsize=12)
    ax[1].legend(fontsize=10)
    ax[1].grid(True, alpha=0.3)
    
    # Set overall title
    # fig.suptitle(title, fontsize=13, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure
    clean_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    output_path = os.path.join(output_dir, f"plot_{clean_title}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph generated : {output_path}")
    plt.close()

# --- MAIN ---
if __name__ == "__main__":
    # 1. Lancer toutes les simulations
    # Note : j'ai rajouté explicitement tous les paramètres dans chaque scénario 
    # pour éviter les erreurs de recherche de fichiers.
    # COMMENTED OUT: Simulations are already run, using existing CSV data
    # for sc in scenarios:
    #     run_simulation(sc)

    run_simulation(scenarios[0])
        
    print("\nGenerating plots from existing CSV data...")

    # Analysis (1): Impact of the Discount Factor (Gamma)
    plot_comparison(
        title="Impact of the Discount Factor (Gamma)", 
        scenario_ids=["Baseline", "Gamma_1.0", "Gamma_0.5"],
        output_dir="plots"
    )
    
    # Analysis (2): Impact of the Replay Buffer size
    plot_comparison(
        title="Impact of the Replay Buffer size", 
        scenario_ids=["Buffer_500", "Buffer_1000", "Buffer_20000", "Buffer_50000"],
        output_dir="plots"
    )
    
    # Analysis (3): Impact of the number of episodes
    plot_comparison(
        title="Impact of the number of episodes", 
        scenario_ids=["Episodes_200", "Episodes_400", "Episodes_800", "Episodes_1000"],
        output_dir="plots"
    )
    
    # COMMENTED OUT: Separate steps plot for Baseline (now included in main plots)
    # plot_comparison(
    #     title="Evolution of the steps (Steps) - Baseline", 
    #     scenario_ids=["Baseline"],
    #     metric='steps',
    #     ylabel='Steps'
    # )

        