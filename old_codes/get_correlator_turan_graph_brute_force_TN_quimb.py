#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marcin Plodzien
"""
 
 
import time
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


import quimb as qu
import quimb.tensor as qtn

# --- 1. Quantum Constants and Operators ---

sigma_x = np.array([[0, 1.], [1., 0]])
sigma_y = 1j * np.array([[0, -1.], [1., 0]])
sigma_z = np.array([[1., 0], [0, -1.]])


sigma_plus_z = 0.5 * (sigma_x + 1j * sigma_y) # 0.5 * (X + iY)
sigma_plus_x = 0.5 * (sigma_y + 1j * sigma_z) # 0.5 * (Y + iZ)
sigma_plus_y = 0.5 * (sigma_z + 1j * sigma_x) # 0.5 * (Z + iX)

# Lookup dictionary for sigma-plus operators
SIGMA_PLUS_OPS = {
    'z': qu.qu(sigma_plus_z),
    'x': qu.qu(sigma_plus_x),
    'y': qu.qu(sigma_plus_y),
}

def get_sigma_plus(direction):
    """Returns the quimb object for the generalized sigma-plus operator."""
    return SIGMA_PLUS_OPS[direction]

# --- 2. Turán Graph Layout Function ---

def create_turan_graph_layout(n, r, horizontal_rotation_angle=15, vertical_rotation_angle=15, 
                              horizontal_tilt=0.5, vertical_spacing=1.0, group_spacing=5):
    """
    Creates a custom 2D layout for a Turán graph T(n, r) by arranging the 
    partitions in a grid pattern with internal rotation/tilt.
    """
    # Create the Turán graph (complete r-partite graph)
    G = nx.turan_graph(n, r)
    
    # Determine the size of each partition (as equal as possible)
    partition_sizes = [n // r + (1 if x < n % r else 0) for x in range(r)]
    
    # Generate partitions (list of nodes in each partition)
    partitions = []
    current_node = 0
    for size in partition_sizes:
        partitions.append(list(range(current_node, current_node + size)))
        current_node += size

    pos = {}
    grid_size = int(np.ceil(np.sqrt(r))) # Arrange partitions in a square grid

    # Convert rotation angles to radians
    horizontal_radians = np.radians(horizontal_rotation_angle)
    vertical_radians = np.radians(vertical_rotation_angle)

    for i, partition in enumerate(partitions):
        # Calculate the grid position for this group
        row = i // grid_size
        col = i % grid_size
        group_center_x = col * group_spacing
        group_center_y = -row * group_spacing

        # Arrange nodes in a tilted and rotated line around the group's center
        for j, node in enumerate(partition):
            # Original unrotated positions (linear)
            x = j * horizontal_tilt
            y = -j * vertical_spacing
            
            # Apply horizontal rotation
            rotated_x = x * np.cos(horizontal_radians) - y * np.sin(horizontal_radians)
            rotated_y = x * np.sin(horizontal_radians) + y * np.cos(horizontal_radians)
            
            # Apply vertical rotation (2D projection effect)
            final_x = rotated_x * np.cos(vertical_radians)
            final_y = rotated_y
            
            # Set final position relative to the group's center
            pos[node] = (group_center_x + final_x, group_center_y + final_y)
    
    return pos, G

 

 
LAYOUT_PARAMS = {
    'horizontal_rotation_angle': 20, 
    'vertical_rotation_angle': 130,
    'horizontal_tilt': 0.3, 
    'vertical_spacing': 1, 
    'group_spacing': 5
}

 
r = 4   # Number of partitions
m = 3   # Nodes in each group (L = m * r = 24)
n = m * r
L = n   # Total number of nodes/qubits

# Generate graph and layout
pos, graph = create_turan_graph_layout(n, r, **LAYOUT_PARAMS)
K = len(graph.edges()) # Number of edges (needed for final plot)

fig, ax = plt.subplots(1, 1, figsize=(11, 10))
ax.set_title(f"Turán Graph T(L={L}, r={r})")
nx.draw(graph, pos, node_size=80, alpha=1, node_color="blue", edge_color="black", with_labels=False)
plt.savefig(f"./fig_turan_graph_L.{L}_r.{r}.png", format="png", dpi=500, bbox_inches='tight')
plt.close(fig) # Close figure 1

# --- Block 2: 2x2 Subplots of Turán Graphs ---
fig_sub, axes = plt.subplots(2, 2, figsize=(12, 12))
plt.subplots_adjust(bottom=0.1, right=0.99, top=0.9)

 
graph_configs = [
    (2, 6, axes[0, 0], "r=2, m=6 (L=12)"),
    (3, 6, axes[0, 1], "r=3, m=6 (L=18)"),
    (4, 3, axes[1, 0], "r=4, m=3 (L=12)"),
    (6, 2, axes[1, 1], "r=6, m=2 (L=12)"),
]

for r_val, m_val, ax_sub, title in graph_configs:
    n_val = r_val * m_val
    pos_sub, graph_sub = create_turan_graph_layout(n_val, r_val, **LAYOUT_PARAMS)
    
    nx.draw(graph_sub, pos_sub, node_size=80, alpha=1, node_color="blue", edge_color="black", with_labels=False, ax=ax_sub)
    ax_sub.set_title(title)

plt.show()
plt.close(fig_sub) 

# --- 4. Quantum Cluster State Preparation ---
 
print(f"L = {L}")

# Map graph node labels to circuit qubit indices
nodes_to_qubit = {node: idx for idx, node in enumerate(graph.nodes())}

# Create the quantum circuit
cluster_state = qtn.Circuit(L)

# Apply Hadamard gates to all qubits: H|0>^L
for qubit_i in range(0, L):
    cluster_state.apply_gate('H', qubit_i)
    
# Apply CZ gates for every edge (Creates the Cluster State |C_G>)
for edge in graph.edges:
    node_i = nodes_to_qubit[edge[0]]
    node_j = nodes_to_qubit[edge[1]]
    # print(node_i, node_j) # Original print statement
    cluster_state.apply_gate('CZ', node_i, node_j)

print(f"Cluster State on T({L},{r}) initialized.")


# --- 5. Entanglement Sweep Configuration ---

# Directions to test for the sigma-plus operator
sigma_plus_vec = ["z", "x"]

# Generate all possible combinations (product space)
all_sigma_plus_sets = list(itertools.product(sigma_plus_vec, repeat=L))

data_optimized_correlator = []

 
NORM_ENT = 4**L
NORM_BELL = 2**L

print(f"\n--- Starting Sweep of {len(all_sigma_plus_sets)} Operator Sets ---")

# --- 6. Entanglement Sweep Loop ---

for idx, sigma_plus_set in enumerate(all_sigma_plus_sets):
    start_time = time.time()
    
    # Apply the product operator M = prod_i sigma^+_{k_i}
    cluster_state_ = cluster_state.copy()
    for qubit_i in range(0, L):
        direction = sigma_plus_set[qubit_i]
        cluster_state_.apply_gate(get_sigma_plus(direction), qubit_i)
        
    # Calculate the inner product: <C_G | M | C_G>
    inner_product = cluster_state.psi.H @ cluster_state_.psi

    Epsilon = np.abs(inner_product)**2
    
    # Calculate measures using pre-calculated factors
    NEpsilon_ent = NORM_ENT * Epsilon
    NEpsilon_bell = NORM_BELL * Epsilon
    
    # Q_ent = log_4(N_ent * Epsilon)
    Q_ent = np.emath.logn(4, NEpsilon_ent).real
    
    # Q_bell = log_2(N_bell * Epsilon)
    Q_bell = np.emath.logn(2, NEpsilon_bell).real
    
 
    gamma_calc = L - (Q_ent + 1)
    
    # Store local data dictionary
    data_dict_local = {
        "L": L,
        "Q_ent": Q_ent, 
        "Q_bell": Q_bell,
        "sigma_plus_set": sigma_plus_set,
        # Store gamma explicitly for completeness, even if the definition 
        # is used later in the original code's output.
        "gamma": gamma_calc
    }
    data_optimized_correlator.append(data_dict_local)
    
 
    stop_time = time.time()
    duration = (stop_time - start_time) / 60
    
 
    string = "TN | L = " + str(L) + " | " + str(idx) + "/" + str(len(all_sigma_plus_sets)) + " "
    string = string + "".join(sigma_plus_set) + "\n"
    string = string + " | Q_ent = " + "{:2.5f}".format(Q_ent)
    string = string + " | Q_bell = " + "{:2.5f}".format(Q_bell)
    print("Duration : {:2.2f} [m]".format(duration))
    print(string)
    
 
data_optimized_correlator = pd.DataFrame(data_optimized_correlator)

# --- 7. Final Analysis and Plotting ---

# Find maximum Q_ent value
Q_ent_max = data_optimized_correlator["Q_ent"].max()

# Filter DataFrame to show all sets that achieved max Q_ent
data_Q_ent_max = data_optimized_correlator[
    data_optimized_correlator["Q_ent"] == Q_ent_max
].copy() # Use .copy() to avoid SettingWithCopyWarning

 
gamma_for_title = L - (Q_ent_max + 1)

# Format the optimal directions summary string
string_optimal_directions = ""
for _, row in data_Q_ent_max.iterrows():
    Q_bell = row["Q_bell"]
    Q_ent = row["Q_ent"]
    set_str = "".join(row["sigma_plus_set"]).lower()
    
    string_optimal_directions += (
        set_str + r"$ | Q_{ent} = $" + "{:2.2f}".format(Q_ent) + 
        r" | $Q_{bell} = $" + "{:2.2f}".format(Q_bell) + "\n"
    )

print("=============")
print("Optimal Directions (Max Q_ent):")
print(string_optimal_directions)



fig_final, ax_final = plt.subplots(1, 1, figsize=(10, 10))


title_string = "TN | L = " + str(L) + " | edges: K = " + str(K) + " | groups: r = " + str(r) + " | nodes in group: m = " + str(m)
title_string = title_string + r" | $\gamma = $" + "{:2.2f}".format(gamma_for_title) + "\n"
title_string = title_string + " | # = " + str(len(data_Q_ent_max)) + " | Optimal Sets: \n"
title_string = title_string + string_optimal_directions


ax_final.set_title(title_string, fontsize=12)
nx.draw(graph, pos, with_labels=True, node_color='lightblue', ax=ax_final)


output_filename = f"./figures_turan_graphs/fig_TN_turan_graph_L.{L}_r.{r}_m.{m}.png"
# plt.savefig(output_filename, format="png", dpi=200)
# print(f"\nFinal graph visualization saved to {output_filename}")
plt.show()
plt.close(fig_final)
