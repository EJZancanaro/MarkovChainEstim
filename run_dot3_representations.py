#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script FINAL - Heatmaps avec FONT SIZE AUTO-ADAPTATIVE
✅ Taille police automatique selon nb états / digits
✅ Évite superposition chiffres
✅ square=False si trop petit
"""

import argparse
import markovchain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Chaîne de Markov + visualisations")
    parser.add_argument("-i", "--input", required=True, help="Fichier RMSD CSV")
    parser.add_argument("--plot-heatmaps", action="store_true", help="Heatmaps")
    parser.add_argument("--plot-dot", action="store_true", help="DOT meilleure")
    parser.add_argument("--plot-dot-all", action="store_true", help="DOT toutes")
    parser.add_argument("-d", "--digits", type=int, default=2, 
                       help="Nombre décimales (défaut: 2)")
    return parser.parse_args()

def get_optimal_fontsize(n_states, digits):
    """
    Calcule taille police optimale :
    - Gros états → petite police
    - Beaucoup digits → petite police
    """
    base_size = 12
    size_penalty = (n_states - 2) * 1.5 + digits * 0.8
    fontsize = max(6, base_size - size_penalty)  # Min 6pt
    return fontsize

def plot_heatmaps(lower_data_dict, upper_data_dict, methods, state_space, digits):
    n_methods = len(methods)
    n_states = len(state_space)
    
    # Taille police optimale
    fontsize = get_optimal_fontsize(n_states, digits)
    print(f"📏 Police heatmaps: {fontsize}pt (états:{n_states}, digits:{digits})")
    
    # Figure plus large si beaucoup méthodes
    fig_width = max(4, 5 * n_methods)
    fig, axes = plt.subplots(2, n_methods, figsize=(fig_width, 10))
    if n_methods == 1: axes = axes.reshape(2, 1)

    fmt_str = f'.{digits}f'

    for idx, method in enumerate(methods):
        lower_matrix = pd.DataFrame(
            lower_data_dict[method].reshape(n_states, n_states),
            index=state_space, columns=state_space
        )
        upper_matrix = pd.DataFrame(
            upper_data_dict[method].reshape(n_states, n_states),
            index=state_space, columns=state_space
        )

        # HEATMAP INF
        ax_l = axes[0, idx] if n_methods > 1 else axes[0]
        sns.heatmap(
            lower_matrix, 
            annot=True, 
            cmap='Blues', 
            fmt=fmt_str,
            ax=ax_l,
            square=(n_states <= 4),  # Carré seulement si petit
            annot_kws={'fontsize': fontsize, 'weight': 'bold'},
            cbar_kws={'shrink': 0.8},
            linewidths=0.5
        )
        ax_l.set_title(f'{method}\nBorne INF', fontsize=fontsize+1, pad=10)

        # HEATMAP SUP
        ax_u = axes[1, idx] if n_methods > 1 else axes[1]
        sns.heatmap(
            upper_matrix, 
            annot=True, 
            cmap='Reds', 
            fmt=fmt_str,
            ax=ax_u,
            square=(n_states <= 4),
            annot_kws={'fontsize': fontsize, 'weight': 'bold'},
            cbar_kws={'shrink': 0.8},
            linewidths=0.5
        )
        ax_u.set_title(f'{method}\nBorne SUP', fontsize=fontsize+1, pad=10)

    plt.suptitle(f'Matrices de confiance (.{digits}d)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'heatmaps_d{digits}_{n_states}s.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    plt.close()
    print(f"✅ heatmaps_d{digits}_{n_states}s.png (font:{fontsize}pt)")

def matrix_to_dot(lower_matrix, upper_matrix, state_space, method, fname, digits):
    g = Digraph(format='png')
    g.attr('node', shape='circle', fillcolor='lightblue')
    g.attr('edge', fontsize=str(9 + digits))
    
    for i in state_space:
        for j in state_space:
            low = lower_matrix.loc[i,j]
            up = upper_matrix.loc[i,j]
            if low > 0:
                label = f"{low:.{digits}f}-{up:.{digits}f}"
                width = str(max(1, (low+up)/2 * 5))
                color = 'green' if (up-low) < 0.1 else 'red'
                g.edge(i, j, label=label, penwidth=width, color=color)
    
    g.render(fname, cleanup=True)
    print(f"✅ {fname}.png")

def print_matrix_formatted(matrix, title, digits):
    fmt_str = f'{{:.{digits}f}}'
    formatted = matrix.round(digits).map(fmt_str.format)
    print(f"\n{title}:")
    print(formatted.to_string(float_format=f'%.{digits}f'))

def main():
    args = parse_args()
    digits = args.digits
    print(f"🔢 Format: .{digits}d")

    # Chargement
    df = pd.read_csv(args.input)
    df['Best'] = df.idxmin(axis=1)
    trajectory = df['Best'].tolist()
    
    print(f"📊 Trajectoire ({len(trajectory)}): {trajectory}")

    mc = markovchain.MarkovChain()
    for state in trajectory: 
        mc.next_state(state)

    print("Proportion of states visited:")

    proportions_states = mc.proportions_states()
    print(proportions_states)

    if np.isclose( proportions_states.sum(),1):
        print("The proportions are normalized, as expected")
    else:
        raise ValueError("The proportions of states do not sum to 1")

    state_space = list(mc.state_space)
    print(f"🔹 États ({len(state_space)}): {state_space}")

    methods = ['Gaussian', 'GaussianSlutsky', 'BasicChi2', 
               'BasicSlutskyChi2', 'FreerChi2', 'FreerSlutskyChi2']
    
    lower_data = {}
    upper_data = {}
    norms = []

    print(f"\n{'='*70}")
    print(f"MATRICES (.{digits}d)")
    print('='*70)

    for method in methods:
        lower_m = pd.DataFrame(0.0, index=state_space, columns=state_space)
        upper_m = pd.DataFrame(0.0, index=state_space, columns=state_space)
        
        for i in state_space:
            for j in state_space:
                lower_m.loc[i,j], upper_m.loc[i,j] = mc.confidence_intervals(
                    state_i=i, state_j=j, method=method)
        
        print_matrix_formatted(lower_m, f"{method} - INF", digits)
        
        lower_data[method] = lower_m.values.flatten()
        upper_data[method] = upper_m.values.flatten()
        
        diff_norm = np.linalg.norm((upper_m - lower_m).values.flatten(), np.inf)
        norms.append(diff_norm)
        print(f"   Norme ∞: {diff_norm:.{digits+1}f}")

    best_idx = np.argmin(norms)
    best_method = methods[best_idx]
    print(f"\n🎯 Meilleure: {best_method} (norme: {norms[best_idx]:.{digits}f})")

    # Visualisations
    if args.plot_heatmaps:
        plot_heatmaps(lower_data, upper_data, methods, state_space, digits)
    
    if args.plot_dot or args.plot_dot_all:
        n_states = len(state_space)
        lower_best = pd.DataFrame(
            lower_data[best_method].reshape(n_states, n_states),
            index=state_space, columns=state_space
        )
        upper_best = pd.DataFrame(
            upper_data[best_method].reshape(n_states, n_states),
            index=state_space, columns=state_space
        )
        matrix_to_dot(lower_best, upper_best, state_space, best_method, 
                     f'markov_best_d{digits}', digits)
        
        if args.plot_dot_all:
            os.makedirs('dot_methods', exist_ok=True)
            for m in methods:
                l_m = pd.DataFrame(
                    lower_data[m].reshape(n_states, n_states),
                    index=state_space, columns=state_space
                )
                u_m = pd.DataFrame(
                    upper_data[m].reshape(n_states, n_states),
                    index=state_space, columns=state_space
                )
                fname = f'dot_methods/markov_{m.lower()}_d{digits}'
                matrix_to_dot(l_m, u_m, state_space, m, fname, digits)

    print(f"\n✅ Fini (.{digits}d)")

if __name__ == "__main__":
    main()

