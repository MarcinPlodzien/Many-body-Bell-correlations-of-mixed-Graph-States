#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marcin Plodzien
"""

import itertools

import numpy as np
import pandas as pd
import torch as pt
from torch import matrix_exp as expm
import networkx as nx
import matplotlib.pyplot as plt

 

id_local = pt.tensor([[1.0, 0.0], [0.0, 1.0]]) + 0j
sigma_x = pt.tensor([[0.0, 1.0], [1.0, 0.0]]) + 0j
sigma_y = 1j * pt.tensor([[0.0, -1.0], [1.0, 0.0]]) + 0j
sigma_z = pt.tensor([[1.0, 0.0], [0.0, -1.0]]) + 0j

hadamard = (
    1.0 / pt.sqrt(pt.tensor(2.0)) * pt.tensor([[1.0, 1.0], [1.0, -1.0]]) + 0j
)

 

def get_string_operator(A: pt.Tensor, L: int, site: int) -> pt.Tensor:
    """
    Embed a single-site operator A on site `site` in an L-site chain
    (1-based index), using tensor products with identity elsewhere.
    """
    if site < 1 or site > L:
        raise ValueError(f"site must be in [1, {L}], got {site}")

    if L == 1:
        return A

    # Left identities
    if site == 1:
        Id = id_local
        for _ in range(2, L):
            Id = pt.kron(Id, id_local)
        return pt.kron(A, Id)

    if site == L:
        Id = id_local
        for _ in range(2, L):
            Id = pt.kron(Id, id_local)
        return pt.kron(Id, A)

    # 1 < site < L
    Id_left = id_local
    for _ in range(1, site - 1):
        Id_left = pt.kron(Id_left, id_local)

    Id_right = id_local
    for _ in range(site + 1, L):
        Id_right = pt.kron(Id_right, id_local)

    return pt.kron(Id_left, pt.kron(A, Id_right))


def get_identity_power(k: int) -> pt.Tensor:
    """Return k-fold tensor product of the local identity: (I ⊗ ... ⊗ I)."""
    if k < 1:
        raise ValueError("k must be >= 1")

    Id = id_local
    for _ in range(1, k):
        Id = pt.kron(Id, id_local)
    return Id


def get_string_two_qubit_operator(A: pt.Tensor, L: int, site: int) -> pt.Tensor:
    """
    Embed a two-qubit operator A acting on (site, site+1) in an L-site chain
    (1-based indexing for site).
    """
    if L == 2:
        return A

    if site < 1 or site > L - 1:
        raise ValueError(f"site must be in [1, {L-1}], got {site}")

    if site == 1:
        return pt.kron(A, get_identity_power(L - 2))

    if site == L - 1:
        return pt.kron(get_identity_power(L - 2), A)

    # 1 < site < L - 1
    left = get_identity_power(site - 1)
    right = get_identity_power(L - (site + 1))
    return pt.kron(left, pt.kron(A, right))


def get_expectation_value(psi: pt.Tensor, A: pt.Tensor) -> pt.Tensor:
    """Return ⟨psi | A | psi⟩ (real part)."""
    exp_val = pt.vdot(psi, A @ psi)
    return exp_val.real


def generate_basis_via_sigma_z(L: int, Z_dict: dict[int, pt.Tensor]) -> pt.Tensor:
    """
    Generate a basis in terms of σ_z eigenvalues for all L sites.
    Returns an array of shape (2**L, L) with entries in {0, 1},
    where 1 corresponds to eigenvalue +1, 0 to -1.
    """
    D = 2 ** L
    basis = pt.zeros((D, L), dtype=pt.float64)

    for v in range(D):
        fock_state = pt.zeros(D, dtype=pt.complex64)
        fock_state[v] = 1.0 + 0j

        for site in range(1, L + 1):
            basis[v, site - 1] = get_expectation_value(fock_state, Z_dict[site]).real

    # Map {-1, +1} -> {0, 1}
    return (basis + 1.0) / 2.0


# -----------------------------------------------------------------------------
# Correlators (entanglement / Bell)
# -----------------------------------------------------------------------------

def get_correlator(
    psi: pt.Tensor,
    sigma_plus_pattern: tuple[str, ...],
    L: int,
    X_dict: dict[int, pt.Tensor],
    Y_dict: dict[int, pt.Tensor],
    Z_dict: dict[int, pt.Tensor],
) -> tuple[float, float]:
    """
    Calculate many-body entanglement and Bell correlators as in
    https://arxiv.org/abs/2406.10662

    sigma_plus_pattern: tuple of "X", "Y", "Z" of length L.
    """
    psi_tmp = psi.clone()

    for i in range(L, 0, -1):
        axis = sigma_plus_pattern[i - 1]

        if axis == "Z":
            sigma_plus_i = 0.5 * (X_dict[i] + 1j * Y_dict[i])
        elif axis == "X":
            sigma_plus_i = 0.5 * (Y_dict[i] + 1j * Z_dict[i])
        elif axis == "Y":
            sigma_plus_i = 0.5 * (Z_dict[i] + 1j * X_dict[i])
        else:
            raise ValueError(f"Invalid axis {axis}, must be one of 'X', 'Y', 'Z'.")

        psi_tmp = sigma_plus_i @ psi_tmp

    E_opt = pt.abs(pt.vdot(psi_tmp, psi)) ** 2

    N_E_opt_bell = E_opt * (2 ** L)
    N_E_opt_ent = E_opt * (4 ** L)

    Q_opt_ent = np.emath.logn(4, N_E_opt_ent)  # entanglement correlator
    Q_opt_bell = np.emath.logn(2, N_E_opt_bell)  # Bell correlator

    return float(np.around(Q_opt_ent, 2)), float(np.around(Q_opt_bell, 2))


# -----------------------------------------------------------------------------
# Two-qubit gates
# -----------------------------------------------------------------------------

def get_CNOT(
    control: int,
    target: int,
    Id: pt.Tensor,
    Z_dict: dict[int, pt.Tensor],
    X_dict: dict[int, pt.Tensor],
) -> pt.Tensor:
    """
    Many-body CNOT with control and target sites (1-based),
    implemented via exp(i π / 4 (I - Z_c)(I - X_t)).
    """
    if control == target:
        return Id

    return expm(1j * pt.pi * (Id - Z_dict[control]) @ (Id - X_dict[target]) * 0.25)


def get_CZ(
    control: int,
    target: int,
    Hadamard_dict: dict[int, pt.Tensor],
    Id: pt.Tensor,
    Z_dict: dict[int, pt.Tensor],
    X_dict: dict[int, pt.Tensor],
) -> pt.Tensor:
    """Many-body controlled-Z via H_t CNOT(c, t) H_t."""
    return (
        Hadamard_dict[target]
        @ get_CNOT(control, target, Id, Z_dict, X_dict)
        @ Hadamard_dict[target]
    )


 
def main():
    # Number of leaves in the star graph (center is node 0)
    N = 8

    # Star graph with center node 0 and leaves 1..N
    graph = nx.star_graph(N)

 
    L = graph.number_of_nodes()
    print(f"L = {L}")

    D = 2 ** L

 
    Id = get_string_operator(id_local, L, 1)

 
    X = {}
    Y = {}
    Z = {}
    Hadamard_dict = {}

    for site in range(1, L + 1):
        X[site] = get_string_operator(sigma_x, L, site).to_sparse_coo()
        Y[site] = get_string_operator(sigma_y, L, site).to_sparse_coo()
        Z[site] = get_string_operator(sigma_z, L, site).to_sparse_coo()
        Hadamard_dict[site] = get_string_operator(hadamard, L, site).to_sparse_coo()

 
    basis = generate_basis_via_sigma_z(L, Z)
  
    psi_ini = pt.ones(D, dtype=pt.complex64) + 0j
    psi_ini = psi_ini / pt.sqrt(pt.vdot(psi_ini, psi_ini))

 
    sigma_plus_vec = ("Z", "X")
    all_sigma_plus_sets = list(itertools.product(sigma_plus_vec, repeat=L))

 
    for i in range(1, N + 1):
 
        if i + 1 <= N:
            j = i + 1
        else:
            j = 1
        graph.add_edge(i, j)

        # Only draw leaves in a circle, with center at 0
        edge_nodes = [node for node in graph.nodes if node != 0]
        pos = nx.circular_layout(graph.subgraph(edge_nodes))
        center_node = 0
        pos[center_node] = np.array([0.0, 0.0])

        # Apply CZ along all edges
        psi_graph = psi_ini.clone()
        for node_u, node_v in graph.edges:
            node_i = node_u + 1  # map graph node label (0..N) -> site index (1..L)
            node_j = node_v + 1

            # Site indices must be in [1, L]
            if not (1 <= node_i <= L) or not (1 <= node_j <= L):
                raise RuntimeError(f"Invalid site index: ({node_i}, {node_j})")

            psi_graph = get_CZ(
                node_i, node_j, Hadamard_dict, Id, Z, X
            ) @ psi_graph

        psi = psi_graph

        # Scan over all sigma_plus patterns, find optimal correlators
        data_optimized_correlator = []
        for sigma_plus_set in all_sigma_plus_sets:
            Q_opt_ent, Q_opt_bell = get_correlator(psi, sigma_plus_set, L, X, Y, Z)
            data_optimized_correlator.append(
                {
                    "L": L,
                    "Q_ent": Q_opt_ent,
                    "Q_bell": Q_opt_bell,
                    "sigma_plus_set": sigma_plus_set,
                }
            )

            s = " L = " + str(L) + " "
            s += "".join(sigma_plus_set) + "\n"
            s += " | Q_ent = " + "{:2.5f}".format(Q_opt_ent)
            s += " | Q_bell = " + "{:2.5f}".format(Q_opt_bell)
            print(s)

        data_optimized_correlator = pd.DataFrame(data_optimized_correlator)

        Q_ent_max = data_optimized_correlator["Q_ent"].max()
        data_Q_ent_max = data_optimized_correlator[
            data_optimized_correlator["Q_ent"] == Q_ent_max
        ]

        string_optimal_directions = ""
        for _, row in data_Q_ent_max.iterrows():
            Q_bell = row["Q_bell"]
            Q_ent = row["Q_ent"]
            string_optimal_directions += "".join(row["sigma_plus_set"]).lower()
            string_optimal_directions += (
                r"$ | Q_{ent} = $" + str(Q_ent) + " | $Q_{bell} = $" + str(Q_bell) + "\n"
            )

        print(string_optimal_directions)

        title_string = f"L = {L} | # = {len(data_Q_ent_max)}\n"
        title_string += string_optimal_directions

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_title(title_string, fontsize=16)
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", ax=ax)
        plt.show()


if __name__ == "__main__":
    main()
