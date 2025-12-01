#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Data Visualizer (Top-Bottom Layout)
===========================================

- Top Panel: Graph Topology (Banner style)
- Bottom Panel: Physics Results (Bell Correlations)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# ==============================================================================
# 1. CONFIGURATION & STYLES
# ==============================================================================
RESULTS_DIR = "results"

# Global Plot Settings
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family'] = 'sans-serif' 

# --- METHOD STYLES ---
METHOD_STYLES = {
    'DM': {'marker': 'o', 'ms': 9, 'mfc': 'white', 'mew': 2, 'ls': '-', 'zorder': 3, 'label': 'Exact DM'},
    'TN': {'marker': 'o', 'ms': 5, 'mfc': 'auto', 'mew': 0, 'ls': '--', 'zorder': 4, 'label': 'Tensor Net'},
    'MC': {'marker': 'x', 'ms': 6, 'mfc': 'none', 'mew': 2, 'ls': ':', 'zorder': 5, 'label': 'Monte Carlo'}
}

# --- COLORS ---
NOISE_COLORS = {
    "depolarizing": "tab:blue",
    "bit_flip": "tab:orange",
    "phase_flip": "tab:green",
    "phase_damping": "tab:green",
    "amplitude_damping": "tab:red"
}

GRAPH_COLORS = {
    'nodes_center': '#FFD700',  # Gold
    'nodes_leaf':   '#87CEEB',  # SkyBlue
    'edge_lines':   '#555555',  # Dark Gray
    'grid':         '#EAEAEA',  # Very light gray
}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def load_data(N, graph_type, noise_type):
    if not os.path.exists(RESULTS_DIR): return None
    pattern = os.path.join(RESULTS_DIR, f"data_N{N}_{graph_type}_{noise_type}_methods-*.pkl")
    files = glob.glob(pattern)
    if not files: return None
    return pd.read_pickle(max(files, key=os.path.getctime))

def draw_topology(ax, N, graph_type):
    """
    Draws the topology in a 'Banner' style (Horizontal).
    """
    ax.axis('off') 
    # crucial for top-plot to ensure circles don't become ovals
    ax.set_aspect('equal') 
    
    # 1. Create Graph & Layouts
    if graph_type == "star":
        G = nx.star_graph(N - 1)
        # Symmetrical layout
        pos = {0: np.array([0, 0])}
        radius = 1.0
        for i in range(1, N):
            angle = 2 * np.pi * (i-1) / (N-1) + (np.pi/2)
            pos[i] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            
    elif graph_type in ["chain", "linear"]:
        G = nx.path_graph(N)
        # Horizontal layout for Top-Banner
        pos = {i: np.array([i, 0]) for i in range(N)} 
        
    elif graph_type in ["cycle", "ring"]:
        G = nx.cycle_graph(N)
        pos = nx.circular_layout(G)
    else:
        G = nx.complete_graph(N)
        pos = nx.spring_layout(G)

    # 2. Draw Edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.5, 
                           edge_color=GRAPH_COLORS['edge_lines'], alpha=0.8)
    
    # 3. Draw Nodes
    if 0 in pos:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[0], node_size=800, 
                               node_color=GRAPH_COLORS['nodes_center'], 
                               edgecolors='#333333', linewidths=2.5)
    leaves = [n for n in G.nodes() if n != 0]
    if leaves:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=leaves, node_size=600, 
                               node_color=GRAPH_COLORS['nodes_leaf'], 
                               edgecolors='#333333', linewidths=2.0)
    
    # 4. Labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight='bold')
    
    # Title for the topology section
    ax.set_title(f"System Topology: {graph_type.capitalize()} Graph (N={N})", 
                 fontsize=14, color='#444444', fontweight='bold', pad=10)

# ==============================================================================
# 3. MASTER PLOTTER
# ==============================================================================

def plot_physics_comparison(N, graph_type, noise_types):
    print("\n--- Generating Top-Bottom Master Plot ---")
    
    # LAYOUT: 2 Rows, 1 Column. 
    # height_ratios=[1, 3] -> Top is small header, Bottom is main plot
    fig, (ax_topo, ax_data) = plt.subplots(2, 1, figsize=(10, 11), 
                                           gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.15})
    
    # --- TOP PANEL: TOPOLOGY ---
    draw_topology(ax_topo, N, graph_type)

    # --- BOTTOM PANEL: DATA ---
    active_noises = []     
    active_methods = set() 

    for noise in noise_types:
        df = load_data(N, graph_type, noise)

        # --- MOCK DATA (Remove when using real files) ---
        if df is None:
            p = np.linspace(0, 0.15, 8)
            base_Q = (N-1) * np.exp(-p*10)
            df = pd.DataFrame({
                'p_noise': p,
                'dm_Q': base_Q,
                'tn_Q': base_Q - 0.05*p,
                'mc_Q': base_Q + 0.1*np.random.normal(0, 0.2, len(p))*p,
                'mc_Q_err': 0.15 * p + 0.05
            })
            if N > 10: df['dm_Q'] = np.nan
        # ------------------------------------------------

        color = NOISE_COLORS.get(noise, 'gray')
        active_noises.append((noise, color))
        
        # PLOT: Exact DM
        if not df['dm_Q'].isna().all():
            active_methods.add('DM')
            st = METHOD_STYLES['DM']
            ax_data.plot(df['p_noise'], df['dm_Q'], color=color, 
                         marker=st['marker'], ms=8, mfc='white', mew=2,
                         ls='-', lw=2.5, alpha=0.8, zorder=3)

        # PLOT: Tensor Net
        if not df['tn_Q'].isna().all():
            active_methods.add('TN')
            st = METHOD_STYLES['TN']
            ax_data.plot(df['p_noise'], df['tn_Q'], color=color, 
                         marker='o', ms=4, mfc=color, mew=0,
                         ls='--', lw=2, zorder=4)

        # PLOT: Monte Carlo
        if not df['mc_Q'].isna().all():
            active_methods.add('MC')
            ax_data.errorbar(df['p_noise'], df['mc_Q'], yerr=df['mc_Q_err'], 
                             fmt='none', ecolor=color, elinewidth=1.5, capsize=3, zorder=5)
            ax_data.plot(df['p_noise'], df['mc_Q'], color=color,
                         marker='x', ms=6, lw=0, zorder=5)

    # --- STYLING DATA PLOT ---
    ax_data.grid(True, color=GRAPH_COLORS['grid'], linewidth=1)
    ax_data.set_axisbelow(True)
    
    ax_data.axhline(0, color='#666666', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_data.text(0.0, 0.05, " Separable Bounds (Q=0)", transform=ax_data.get_yaxis_transform(), 
                 color='#666666', fontsize=10, va='bottom')

    # Clean Spines
    ax_data.spines['top'].set_visible(False)
    ax_data.spines['right'].set_visible(False)
    ax_data.spines['left'].set_linewidth(1.5)
    ax_data.spines['bottom'].set_linewidth(1.5)

    ax_data.set_xlabel(r"noise  ($p$)", fontsize=14, labelpad=10)
    ax_data.set_ylabel(r"many-body Bell correlator ($Q$)", fontsize=14, labelpad=10)
    ax_data.set_ylim([-2, N-1.5])

    # --- LEGEND ---
    legend_elements = [Line2D([0], [0], color='w', label=r'$\bf{Noise\ Model}$')]
    for name, color in dict(active_noises).items():
        legend_elements.append(Line2D([0], [0], color=color, lw=3, label=name.replace('_', ' ').title()))
        
    legend_elements.append(Line2D([0], [0], color='w', label=' ')) # Spacer
    legend_elements.append(Line2D([0], [0], color='w', label=r'$\bf{Method}$'))
    
    if 'DM' in active_methods:
        legend_elements.append(Line2D([0], [0], color='k', marker='o', mfc='w', mew=2, label='Exact DM'))
    if 'TN' in active_methods:
        legend_elements.append(Line2D([0], [0], color='k', marker='o', mfc='k', mew=0, ls='--', label='Tensor Net'))
    if 'MC' in active_methods:
        legend_elements.append(Line2D([0], [0], color='k', marker='x', lw=0, mew=2, label='Monte Carlo'))

    # Legend placement: Upper Right of the BOTTOM plot
    ax_data.legend(handles=legend_elements, loc='upper right', frameon=True, 
                   facecolor='white', framealpha=0.9, fontsize=11)

    # Save
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    out_name = f"{RESULTS_DIR}/plot_vertical_N{N}_{graph_type}.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight') 
    print(f"[Done] Saved high-res plot to: {out_name}")
    plt.show()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    N = 12
    graph_type = "star"
    # noise_list = ["depolarizing", "bit_flip", "phase_flip", "amplitude_damping"]
    
    noise_list = ["depolarizing",    "amplitude_damping", "bit_flip"]
    
    plot_physics_comparison(N, graph_type, noise_list)