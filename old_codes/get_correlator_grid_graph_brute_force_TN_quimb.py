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


sigma_x = np.array([[0, 1.], [1., 0]])
sigma_y = 1j * np.array([[0, -1.], [1., 0]])
sigma_z = np.array([[1., 0], [0, -1.]])

# Defining Generalized Sigma-Plus Operators
sigma_plus_z = 0.5 * (sigma_x + 1j * sigma_y)
sigma_plus_x = 0.5 * (sigma_y + 1j * sigma_z)
sigma_plus_y = 0.5 * (sigma_z + 1j * sigma_x)

# Dictionary to map direction strings directly to the quimb operator
SIGMA_PLUS_OPS = {
    'z': qu.qu(sigma_plus_z),
    'x': qu.qu(sigma_plus_x),
    'y': qu.qu(sigma_plus_y),
}

def get_sigma_plus(direction):
    """
    Returns the quimb quantum object ('qu') for the generalized sigma-plus 
    operator corresponding to the given direction using a lookup dictionary.
    """
    return SIGMA_PLUS_OPS[direction]

 
def create_turan_graph_layout(n, r, horizontal_rotation_angle=15, vertical_rotation_angle=15, 
                              horizontal_tilt=0.5, vertical_spacing=1.0, group_spacing=5):
    """Creates a custom layout for a Turán graph (UNUSED)."""
    G = nx.turan_graph(n, r)
    partition_sizes = [n // r + (1 if x < n % r else 0) for x in range(r)]
    partitions = []
    current_node = 0
    for size in partition_sizes:
        partitions.append(list(range(current_node, current_node + size)))
        current_node += size

    pos = {}
    grid_size = int(np.ceil(np.sqrt(r)))
    horizontal_radians = np.radians(horizontal_rotation_angle)
    vertical_radians = np.radians(vertical_rotation_angle)

    for i, partition in enumerate(partitions):
        row = i // grid_size
        col = i % grid_size
        group_center_x = col * group_spacing
        group_center_y = -row * group_spacing

        for j, node in enumerate(partition):
            x = j * horizontal_tilt
            y = -j * vertical_spacing
            rotated_x = x * np.cos(horizontal_radians) - y * np.sin(horizontal_radians)
            rotated_y = x * np.sin(horizontal_radians) + y * np.cos(horizontal_radians)
            final_x = rotated_x * np.cos(vertical_radians)
            final_y = rotated_y
            pos[node] = (group_center_x + final_x, group_center_y + final_y)
    return pos
# ----------------------------------------------------------------------

# --- 1. Graph and Circuit Setup ---

# Dimensions for the grid graph
m = 4  
n = 4  
l = 1  

# Create the grid graph
graph = nx.grid_graph(dim=(n, m, l))

K = len(graph.edges()) # Number of edges
L = len(graph.nodes)   # Number of qubits (nodes)
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
    cluster_state.apply_gate('CZ', node_i, node_j)

 
# Directions to test for the sigma-plus operator on each qubit
sigma_plus_vec = ["z", "x"]

# Generate all possible combinations (product space) of directions
all_sigma_plus_sets = list(itertools.product(sigma_plus_vec, repeat=L))

 
N_batches = 4
batch_size = int(len(all_sigma_plus_sets) / N_batches)
all_sigma_plus_sets_ = all_sigma_plus_sets[3 * batch_size:]
N_batches_ = 3
batch_size_ = int(len(all_sigma_plus_sets_) / N_batches_)
all_sigma_plus_sets_ = all_sigma_plus_sets_[:batch_size_]

print(f"Total combinations generated: {len(all_sigma_plus_sets)}")
print(f"Processing batch of size: {len(all_sigma_plus_sets_)}")

 
data_optimized_correlator = []
counter = 0
start = time.time()
duration = 0
gamma_min = 1000
 
direction_min = all_sigma_plus_sets[1] 

print(f"\n--- Starting Sweep of {len(all_sigma_plus_sets_)} Sets ---")

 
NORM_ENT = 4**L
NORM_BELL = 2**L

for idx, sigma_plus_set in enumerate(all_sigma_plus_sets_):
    
 
    op_list = [get_sigma_plus(d) for d in sigma_plus_set]
 
    cluster_state_ = cluster_state.copy()
    for qubit_i in range(0, L):
        cluster_state_.apply_gate(op_list[qubit_i], qubit_i)
    
 
    inner_product = cluster_state.psi.H @ cluster_state_.psi

    Epsilon = np.abs(inner_product)**2
    
 
    NEpsilon_ent = NORM_ENT * Epsilon
    NEpsilon_bell = NORM_BELL * Epsilon
    
 
    Q_ent = np.emath.logn(4, NEpsilon_ent).real
    Q_bell = np.emath.logn(2, NEpsilon_bell).real
    gamma = (-np.emath.logn(4, Epsilon) - 1).real
    
 
    if(gamma < gamma_min):
        gamma_min = gamma
        direction_min = sigma_plus_set
        
 
    data_dict_local = {
        "L": L,
        "Q_ent": np.around(Q_ent, 2), 
        "Q_bell": np.around(Q_bell, 2),
        "gamma": gamma,
        "sigma_plus_set": sigma_plus_set,
    }
    data_optimized_correlator.append(data_dict_local)
    
 
    string = "TN | L = " + str(L) + " | " + str(idx) + "/" + str(len(all_sigma_plus_sets_)) + " | n = " + str(n) + " | m = " + str(m) + " | l = " + str(l)
    string = string + "".join(sigma_plus_set) + "\n"
    string = string + " | Q_ent = " + "{:2.5f}".format(Q_ent)
    string = string + " | Q_bell = " + "{:2.5f}".format(Q_bell)
    string = string + " | gamma = " + "{:2.2f}".format(gamma) + "\n"
    string = string + " ======"
    string = string + " | gamma_min = " + "{:2.2f}".format(gamma_min) + "\n"
    string = string + "".join(direction_min) + "\n"
    print(string)
    
 
    counter = counter + 1
    if(counter == 1000):
        stop = time.time()
        duration = (stop - start)
        print("Duration : {:2.2f} [s]".format(duration))
        
        total_estimated_time = len(all_sigma_plus_sets_) / 1000 * duration / 3600 / 24
        print("Total time estimation : {:2.2} days".format(total_estimated_time))

 
data_optimized_correlator = pd.DataFrame(data_optimized_correlator)


Q_ent_max = data_optimized_correlator["Q_ent"].max()
data_Q_ent_max = data_optimized_correlator[data_optimized_correlator["Q_ent"] == Q_ent_max]

string_optimal_directions = ""
for idx, row in data_Q_ent_max.iterrows():
    Q_bell = row["Q_bell"]
    Q_ent = row["Q_ent"]
    set_str = "".join(row["sigma_plus_set"]).lower()
    string_optimal_directions += (
        set_str + r" | $Q_{ent} = $" + "{:2.2f}".format(Q_ent) + 
        r" | $Q_{bell} = $" + "{:2.2f}".format(Q_bell) + "\n"
    )

print("=============")
print("Optimal Directions (Max Q_ent):")
print(string_optimal_directions)

 
gamma_for_title = data_Q_ent_max["gamma"].min() if not data_Q_ent_max.empty else 0.0

 
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

title_string = "TN | L = " + str(L) + " | edges: K = " + str(K) + " | n = " + str(n) + " | m = " + str(m) + " | l = " + str(l)
title_string = title_string + r" | $\gamma = $" + "{:2.2f}".format(gamma_for_title) + "\n"
title_string = title_string + " | # = " + str(len(data_Q_ent_max)) + " | Optimal Sets: \n"
title_string = title_string + string_optimal_directions


ax.set_title(title_string, fontsize=12) # Reduced fontsize for complex title
nx.draw(graph,   with_labels=True, node_color='lightblue', ax=ax)
plt.show()
# Save the figure
# plt.savefig("./figure_final.png", dpi=600, format="png")
# print("Final graph visualization saved to ./figure_final.png")
