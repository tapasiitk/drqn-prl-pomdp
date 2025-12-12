import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_stats(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def plot_wsls_comparison(agent_stats, human_stats):
    """
    Bar chart comparing Win-Stay and Lose-Shift probabilities.
    """
    labels = ['Agent', 'Children', 'Adults']
    
    # Extract data
    ws_values = [
        agent_stats['win_stay'], 
        human_stats['Children']['win_stay'], 
        human_stats['Adults']['win_stay']
    ]
    
    ls_values = [
        agent_stats['lose_shift'], 
        human_stats['Children']['lose_shift'], 
        human_stats['Adults']['lose_shift']
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, ws_values, width, label='Win-Stay', color='skyblue')
    rects2 = ax.bar(x + width/2, ls_values, width, label='Lose-Shift', color='salmon')
    
    ax.set_ylabel('Probability')
    ax.set_title('Win-Stay / Lose-Shift Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    # Add values on top
    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    plt.savefig('results/comparison_wsls.png')
    print("Saved results/comparison_wsls.png")
    plt.close()

def plot_reversal_curve(agent_stats, human_stats):
    """
    Line plot of accuracy around the reversal point.
    """
    # X-axis: -5 to +10
    x_axis = np.arange(-5, 11)
    
    # Get curves
    agent_curve = agent_stats['reversal_curve']
    child_curve = human_stats['Children']['reversal_curve']
    adult_curve = human_stats['Adults']['reversal_curve']
    
    # Ensure lengths match (sometimes analysis windows might differ slightly if edited)
    # We slice to the common minimum length just in case
    min_len = min(len(agent_curve), len(child_curve), len(adult_curve), len(x_axis))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_axis[:min_len], agent_curve[:min_len], label='DRQN Agent', linewidth=2, color='blue', marker='o')
    ax.plot(x_axis[:min_len], adult_curve[:min_len], label='Adults', linewidth=2, color='green', linestyle='--')
    ax.plot(x_axis[:min_len], child_curve[:min_len], label='Children', linewidth=2, color='orange', linestyle='--')
    
    ax.axvline(x=0, color='gray', linestyle=':', label='Reversal Point')
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=0.8, color='gray', linestyle='-', alpha=0.3, label='Max Prob (0.8)')
    
    ax.set_xlabel('Trials relative to Reversal')
    ax.set_ylabel('Accuracy (Prob. choosing Correct Option)')
    ax.set_title('Adaptation to Reversal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comparison_reversal_curve.png')
    print("Saved results/comparison_reversal_curve.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_stats", type=str, default="results/agent_stats.pkl")
    parser.add_argument("--human_stats", type=str, default="results/human_stats.pkl")
    args = parser.parse_args()
    
    if not os.path.exists(args.agent_stats) or not os.path.exists(args.human_stats):
        print("Error: Statistics files not found. Run analyze_agent.py and analyze_humans.py first.")
    else:
        print("Loading stats...")
        a_stats = load_stats(args.agent_stats)
        h_stats = load_stats(args.human_stats)
        
        print("Generating plots...")
        plot_wsls_comparison(a_stats, h_stats)
        plot_reversal_curve(a_stats, h_stats)
