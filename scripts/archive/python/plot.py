"""
Publication-Quality Plot Generation for AdaPriv Research Paper.

Generates 7 graphs as specified in experimental_protocol.md:
1. Test Accuracy vs. Training Rounds (CIFAR-10 + FEMNIST)
2. Privacy-Utility Tradeoff Curve
3. Training Loss Convergence
4. Ablation Study (Component Contribution)
5. Per-Round Privacy Budget Evolution
6. Parameter Sensitivity Heatmap
7. Resource Efficiency (Communication Cost)
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re

# Publication-quality settings
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
})

# Color palette (colorblind-friendly)
COLORS = {
    'no-dp': '#0173B2',      # Blue
    'fixed-dp': '#DE8F05',   # Orange
    'adapriv': '#029E73',    # Green
    'device-only': '#CC78BC', # Purple
    'param-only': '#CA9161',  # Brown
    'round-only': '#FBAFE4',  # Pink
}

MARKERS = {
    'no-dp': 'o',
    'fixed-dp': 's',
    'adapriv': '^',
    'device-only': 'D',
    'param-only': 'v',
    'round-only': 'p',
}

LABELS = {
    'no-dp': 'No DP (Baseline)',
    'fixed-dp': 'Fixed DP',
    'adapriv': 'AdaPriv (Ours)',
    'device-only': 'Device-Only',
    'param-only': 'Parameter-Only',
    'round-only': 'Round-Only',
}


def load_experiments(results_dir='./experiment_results'):
    """
    Load all experiment results from JSON files.
    
    Returns:
        Dictionary indexed by (dataset, config, seed, epsilon)
    """
    experiments = defaultdict(list)
    
    pattern = f"{results_dir}/results_*.json"
    files = glob.glob(pattern)
    
    print(f"\n📂 Loading experiments from {results_dir}/")
    print(f"Found {len(files)} result files")
    
    for filepath in files:
        filename = Path(filepath).name
        
        # Parse filename: results_{dataset}_{config}_seed{seed}_eps{epsilon}.json
        match = re.match(r'results_(\w+)_([a-zA-Z0-9_-]+)_seed(\d+)_eps([\d.]+)\.json', filename)
        
        if not match:
            print(f"⚠️  Skipping {filename} (doesn't match pattern)")
            continue
        
        dataset, config, seed, epsilon = match.groups()
        seed = int(seed)
        epsilon = float(epsilon)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        key = (dataset, config, epsilon)
        experiments[key].append({
            'seed': seed,
            'data': data,
            'filepath': filepath
        })
    
    print(f"✅ Loaded {len(experiments)} unique configurations")
    return dict(experiments)


def aggregate_seeds(experiments_list):
    """
    Aggregate results across multiple seeds.
    Fix: index by round number, not list position, to handle early stopping.
    
    Returns:
        Dictionary with mean and std for each metric
    """
    if not experiments_list:
        return None
    
    # Collect metrics across seeds
    metrics_by_round = defaultdict(lambda: defaultdict(list))
    
    for exp in experiments_list:
        rounds = exp['data']['rounds']
        for round_data in rounds:
            rnd = round_data.get('round')
            if rnd is None:
                continue
            for metric, value in round_data.items():
                if isinstance(value, (int, float)) and metric != 'round':  # Only numeric metrics
                    metrics_by_round[rnd][metric].append(value)
    
    # Sort rounds
    sorted_rounds = sorted(metrics_by_round.keys())
    
    # Compute mean and std
    aggregated = {
        'rounds': [],
        'num_seeds': len(experiments_list)
    }
    
    for rnd in sorted_rounds:
        round_agg = {'round': rnd}
        for metric, values in metrics_by_round[rnd].items():
            if values:
                round_agg[f'{metric}_mean'] = np.mean(values)
                round_agg[f'{metric}_std'] = np.std(values)
        aggregated['rounds'].append(round_agg)
    
    return aggregated


def plot_graph1_accuracy_vs_rounds(experiments, output_dir='./figures'):
    """
    Graph 1: Test Accuracy vs. Training Rounds (2 subplots: CIFAR-10 + FEMNIST)
    """
    print("\n📊 Generating Graph 1: Test Accuracy vs. Rounds")
    
    datasets_with_data = []
    for dataset in ['cifar10', 'femnist']:
        for config in ['no-dp', 'fixed-dp', 'adapriv']:
            if (dataset, config, 3.0) in experiments:
                datasets_with_data.append(dataset)
                break
    
    if not datasets_with_data:
        print("  ⏩ Skipping Graph 1: No main comparison data found")
        return

    num_datasets = len(datasets_with_data)
    fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 4), squeeze=False)
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets_with_data):
        ax = axes[idx]
        
        for config in ['no-dp', 'fixed-dp', 'adapriv']:
            key = (dataset, config, 3.0)  # epsilon = 3.0
            
            if key not in experiments:
                print(f"  ⚠️  Missing: {key}")
                continue
            
            agg = aggregate_seeds(experiments[key])
            if not agg:
                continue
            
            rounds = [r['round'] for r in agg['rounds']]
            acc_mean = [r['val_accuracy_mean'] * 100 for r in agg['rounds']]
            acc_std = [r['val_accuracy_std'] * 100 for r in agg['rounds']]
            
            ax.plot(rounds, acc_mean, label=LABELS[config], 
                   color=COLORS[config], marker=MARKERS[config], 
                   markevery=5, linewidth=2)
            ax.fill_between(rounds, 
                           np.array(acc_mean) - np.array(acc_std),
                           np.array(acc_mean) + np.array(acc_std),
                           alpha=0.2, color=COLORS[config])
        
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'({chr(97+idx)}) {dataset.upper()}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    output_path = f"{output_dir}/graph1_accuracy_vs_rounds.png"
    Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def plot_graph2_privacy_utility_tradeoff(experiments, output_dir='./figures'):
    """
    Graph 2: Privacy-Utility Tradeoff Curve
    """
    print("\n📊 Generating Graph 2: Privacy-Utility Tradeoff")
    
    epsilon_values = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    has_data = False
    for config in ['fixed-dp', 'adapriv']:
        for eps in epsilon_values:
            if ('cifar10', config, eps) in experiments:
                has_data = True
                break
    
    if not has_data:
        print("  ⏩ Skipping Graph 2: No epsilon sweep data found")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    epsilon_values = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    
    for config in ['fixed-dp', 'adapriv']:
        epsilons = []
        accs_mean = []
        accs_std = []
        
        for eps in epsilon_values:
            key = ('cifar10', config, eps)
            
            if key not in experiments:
                continue
            
            agg = aggregate_seeds(experiments[key])
            if not agg or not agg['rounds']:
                continue
            
            # Get final accuracy
            final_acc = agg['rounds'][-1]['val_accuracy_mean'] * 100
            final_std = agg['rounds'][-1]['val_accuracy_std'] * 100
            
            epsilons.append(eps)
            accs_mean.append(final_acc)
            accs_std.append(final_std)
        
        ax.errorbar(epsilons, accs_mean, yerr=accs_std, 
                   label=LABELS[config], color=COLORS[config],
                   marker=MARKERS[config], markersize=8,
                   linewidth=2.5, capsize=5, capthick=2)
    
    ax.set_xlabel('Privacy Budget (ε)')
    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Privacy-Utility Tradeoff on CIFAR-10')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    output_path = f"{output_dir}/graph2_privacy_utility_tradeoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def plot_graph3_loss_convergence(experiments, output_dir='./figures'):
    """
    Graph 3: Training Loss Convergence
    """
    print("\n📊 Generating Graph 3: Training Loss Convergence")
    
    has_data = False
    for config in ['no-dp', 'fixed-dp', 'adapriv']:
        if ('cifar10', config, 3.0) in experiments:
            has_data = True
            break
            
    if not has_data:
        print("  ⏩ Skipping Graph 3: No loss convergence data found")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    for config in ['no-dp', 'fixed-dp', 'adapriv']:
        key = ('cifar10', config, 3.0)
        
        if key not in experiments:
            continue
        
        agg = aggregate_seeds(experiments[key])
        if not agg:
            continue
        
        rounds = [r['round'] for r in agg['rounds']]
        loss_mean = [r['avg_loss_mean'] for r in agg['rounds']]
        loss_std = [r['avg_loss_std'] for r in agg['rounds']]
        
        ax.plot(rounds, loss_mean, label=LABELS[config],
               color=COLORS[config], marker=MARKERS[config],
               markevery=5, linewidth=2)
        ax.fill_between(rounds,
                       np.array(loss_mean) - np.array(loss_std),
                       np.array(loss_mean) + np.array(loss_std),
                       alpha=0.2, color=COLORS[config])
    
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Training Loss (log scale)')
    ax.set_title('Training Loss Convergence on CIFAR-10')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    output_path = f"{output_dir}/graph3_loss_convergence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def plot_graph4_ablation_study(experiments, output_dir='./figures'):
    """
    Graph 4: Ablation Study - Component Contribution
    """
    print("\n📊 Generating Graph 4: Ablation Study")
    print("  ⚠️  NOTE: Requires ablation experiments (device-only, param-only, round-only)")
    
    configs = ['device-only', 'param-only', 'round-only', 'adapriv']
    available_configs = [c for c in configs if ('cifar10', c, 3.0) in experiments]
    
    if not available_configs:
        print("  ⏩ Skipping Graph 4: No ablation data found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = ['device-only', 'param-only', 'round-only', 'adapriv']
    accs_mean = []
    accs_std = []
    labels_list = []
    colors_list = []
    
    for config in configs:
        key = ('cifar10', config, 3.0)
        
        if key not in experiments:
            print(f"  ⚠️  Missing: {config}")
            continue
        
        agg = aggregate_seeds(experiments[key])
        if not agg or not agg['rounds']:
            print(f"  ⚠️  Empty data: {config}")
            continue
        
        final_acc = agg['rounds'][-1]['val_accuracy_mean'] * 100
        final_std = agg['rounds'][-1]['val_accuracy_std'] * 100
        
        accs_mean.append(final_acc)
        accs_std.append(final_std)
        labels_list.append(LABELS.get(config, config))
        colors_list.append(COLORS.get(config, '#666666'))
    
    if accs_mean:
        x_pos = np.arange(len(labels_list))
        bars = ax.bar(x_pos, accs_mean, yerr=accs_std, 
                     color=colors_list, alpha=0.8, 
                     capsize=5, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_list, rotation=15, ha='right')
        ax.set_ylabel('Final Test Accuracy (%)')
        ax.set_title('Ablation Study: Component Contribution')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(accs_mean) * 1.2])
        
        # Add value labels on bars
        for bar, val in zip(bars, accs_mean):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    output_path = f"{output_dir}/graph4_ablation_study.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def plot_graph5_epsilon_evolution(experiments, output_dir='./figures'):
    """
    Graph 5: Per-Round Privacy Budget Evolution
    """
    print("\n📊 Generating Graph 5: Per-Round Privacy Budget Evolution")
    
    key = ('cifar10', 'adapriv', 3.0)
    
    if key not in experiments:
        print(f"  ⏩ Skipping Graph 5: No AdaPriv budget data found")
        return
    
    # Use first seed for visualization
    data = experiments[key][0]['data']
    rounds = [r['round'] for r in data['rounds']]
    epsilon_t = [r.get('epsilon_t', 0) for r in data['rounds']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(rounds, epsilon_t, color=COLORS['adapriv'], 
           linewidth=3, marker='o', markersize=6)
    ax.fill_between(rounds, epsilon_t, alpha=0.3, color=COLORS['adapriv'])
    
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Per-Round Privacy Budget (ε_t)')
    ax.set_title('Progressive Privacy Scheduler (Cosine Annealing)')
    ax.grid(True, alpha=0.3)
    
    output_path = f"{output_dir}/graph5_epsilon_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def plot_graph6_sensitivity_heatmap(experiments, output_dir='./figures'):
    """
    Graph 6: Parameter Sensitivity Heatmap
    """
    print("\n📊 Generating Graph 6: Parameter Sensitivity Heatmap")
    
    key = ('cifar10', 'adapriv', 3.0)
    
    if key not in experiments:
        print(f"  ⏩ Skipping Graph 6: No sensitivity heatmap data found")
        return
    
    # Use first seed
    data = experiments[key][0]['data']
    
    # Extract sensitivity data
    sensitivity_matrix = []
    layer_names = None
    
    for round_data in data['rounds']:
        sensitivities = round_data.get('sensitivities', {})
        
        if not sensitivities:
            continue
        
        if layer_names is None:
            layer_names = sorted(sensitivities.keys())
        
        sensitivity_matrix.append([sensitivities.get(layer, 0) for layer in layer_names])
    
    if not sensitivity_matrix:
        print("  ⚠️  No sensitivity data found in results")
        return
    
    sensitivity_matrix = np.array(sensitivity_matrix).T  # Transpose for heatmap
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Shorten layer names for readability
    short_names = [name.split('.')[-1][:15] for name in layer_names]
    
    im = ax.imshow(sensitivity_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Layer')
    ax.set_title('Parameter Sensitivity Evolution Across Training')
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Sensitivity Score (s_j)', rotation=270, labelpad=20)
    
    output_path = f"{output_dir}/graph6_sensitivity_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {output_path}")
    plt.close()


def plot_graph7_communication_cost(experiments, output_dir='./figures'):
    """
    Graph 7: Resource Efficiency - Communication Cost
    """
    print("\n📊 Generating Graph 7: Communication Cost Comparison")
    
    configs = ['no-dp', 'fixed-dp', 'adapriv']
    available_configs = [c for c in configs if ('cifar10', c, 3.0) in experiments]
    
    if not available_configs:
        print("  ⏩ Skipping Graph 7: No communication cost data found")
        return

    avg_resources = []
    std_resources = []
    labels_list = []
    colors_list = []
    
    for config in configs:
        key = ('cifar10', config, 3.0)
        
        if key not in experiments:
            continue
        
        agg = aggregate_seeds(experiments[key])
        if not agg or not agg['rounds']:
            continue
        
        # Average resource score across all rounds
        resources = [r.get('avg_resource_mean', 0.5) for r in agg['rounds']]
        
        avg_resources.append(np.mean(resources))
        std_resources.append(np.std(resources))
        labels_list.append(LABELS[config])
        colors_list.append(COLORS[config])
    
    if avg_resources:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(len(labels_list))
        bars = ax.bar(x_pos, avg_resources, yerr=std_resources,
                     color=colors_list, alpha=0.8,
                     capsize=5, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_list)
        ax.set_ylabel('Average Resource Score')
        ax.set_title('Communication Efficiency (Higher = More Efficient Selection)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])
        
        output_path = f"{output_dir}/graph7_communication_cost.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {output_path}")
        plt.close()


def generate_all_plots(results_dir='./experiment_results', output_dir='./figures'):
    """
    Generate all 7 publication graphs.
    """
    print("\n" + "="*60)
    print("🎨 AdaPriv Publication Plots Generator")
    print("="*60)
    
    experiments = load_experiments(results_dir)
    
    if not experiments:
        print("\n❌ No experiment results found!")
        print(f"   Make sure to run experiments first: ./run_experiments.sh")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate all graphs
    if experiments:
        plot_graph1_accuracy_vs_rounds(experiments, output_dir)
        plot_graph2_privacy_utility_tradeoff(experiments, output_dir)
        plot_graph3_loss_convergence(experiments, output_dir)
        plot_graph4_ablation_study(experiments, output_dir)
        plot_graph5_epsilon_evolution(experiments, output_dir)
        plot_graph6_sensitivity_heatmap(experiments, output_dir)
        plot_graph7_communication_cost(experiments, output_dir)
    else:
        print("\n❌ No experiments to plot.")
    
    print("\n" + "="*60)
    print("✅ All plots generated successfully!")
    print(f"📁 Saved to: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    generate_all_plots()
