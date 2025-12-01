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

 
id_local = pt.tensor([[1.0, 0.0],
                      [0.0, 1.0]], dtype=pt.cdouble)

sigma_x = pt.tensor([[0.0, 1.0],
                     [1.0, 0.0]], dtype=pt.cdouble)

sigma_y = pt.tensor([[0.0, -1.0j],
                     [1.0j, 0.0]], dtype=pt.cdouble)

sigma_z = pt.tensor([[1.0, 0.0],
                     [0.0, -1.0]], dtype=pt.cdouble)

hadamard = (1.0 / pt.sqrt(pt.tensor(2.0))) * pt.tensor([[1.0, 1.0],
                                                        [1.0, -1.0]],
                                                       dtype=pt.cdouble)

 

# ---------------------------------------------------------------------------
# Utility functions for operator strings
# ---------------------------------------------------------------------------

def get_string_operator(A: pt.Tensor, L: int, i: int) -> pt.Tensor:
    """
    Embed a single-qubit operator A at site i (1-based) in an L-qubit chain.
    """
    if not (1 <= i <= L):
        raise ValueError("Site index i must satisfy 1 <= i <= L.")

    if L == 1:
        return A

    if i == 1:
 
        op = A
        for _ in range(2, L + 1):
            op = pt.kron(op, id_local)
        return op

    if i == L:
 
        op = id_local
        for _ in range(2, L):
            op = pt.kron(op, id_local)
        return pt.kron(op, A)

 
    id_left = id_local
    for _ in range(2, i):
        id_left = pt.kron(id_left, id_local)

    id_right = id_local
    for _ in range(i + 1, L):
        id_right = pt.kron(id_right, id_local)

    return pt.kron(id_left, pt.kron(A, id_right))


def get_identity_power(k: int) -> pt.Tensor:
    """Return Identity."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    Id = id_local
    for _ in range(1, k):
        Id = pt.kron(Id, id_local)
    return Id


def get_string_two_qubit_operator(A: pt.Tensor, L: int, i: int) -> pt.Tensor:
    """
    Embed a two-qubit operator A acting on sites (i, i+1) in an L-qubit chain.
    Sites are 1-based, so valid i are 1 <= i <= L-1.
    """
    if L == 2:
        return A

    if not (1 <= i <= L - 1):
        raise ValueError("i must satisfy 1 <= i <= L-1 for a two-qubit operator.")

    if i == 1:
        return pt.kron(A, get_identity_power(L - 2))

    if i == L - 1:
        return pt.kron(get_identity_power(L - 2), A)

    # middle
    left = get_identity_power(i - 1)
    right = get_identity_power(L - (i + 1))
    return pt.kron(left, pt.kron(A, right))


# ---------------------------------------------------------------------------
# Expectation values and basis generation
# ---------------------------------------------------------------------------

def get_expectation_value(psi: pt.Tensor, A: pt.Tensor) -> pt.Tensor:
    """Return ⟨psi|A|psi⟩ (real part)."""
    exp_val = pt.vdot(psi, A @ psi)
    return exp_val.real


def generate_basis_via_sigma_z(L: int, Z: dict) -> pt.Tensor:
    """
    Generate computational basis states labeled by expectation values of Z_i.

    Returns:
        basis (2^L x L) tensor with entries in {0,1}, where
        basis[v, i] = (⟨v|Z_i|v⟩ + 1) / 2.
    """
    D = 2 ** L
    basis = pt.zeros((D, L))
    for v in range(D):
        fock_state = pt.zeros(D, dtype=pt.cdouble)
        fock_state[v] = 1.0
        for i in range(1, L + 1):
            basis[v, i - 1] = get_expectation_value(fock_state, Z[i]).real

    return (basis + 1) / 2


# ---------------------------------------------------------------------------
# Many-body correlator (Ref: https://arxiv.org/abs/2406.10662)
# ---------------------------------------------------------------------------

def get_correlator(
    psi: pt.Tensor,
    sigma_plus_set_initial,
    L: int,
    X: dict,
    Y: dict,
    Z: dict
):
    """
    Calculate many-body entanglement/Bell correlators for a given direction set.

    Args:
        psi: state vector (complex torch tensor of size 2^L).
        sigma_plus_set_initial: iterable of 'X', 'Y' or 'Z' of length L.
        L: number of sites.
        X, Y, Z: dictionaries of string operators (1..L).

    Returns:
        (Q_opt_ent, Q_opt_bell) rounded to 2 decimals.
    """
    psi_tmp = psi.clone()
    for i in np.arange(L, 0, -1):
        axis = sigma_plus_set_initial[i - 1]

        if axis == "Z":
            sigma_plus_i = 0.5 * (X[i] + 1j * Y[i])
        elif axis == "X":
            sigma_plus_i = 0.5 * (Y[i] + 1j * Z[i])
        elif axis == "Y":
            sigma_plus_i = 0.5 * (Z[i] + 1j * X[i])
        else:
            raise ValueError(f"Invalid axis '{axis}', expected one of 'X','Y','Z'.")

        psi_tmp = sigma_plus_i @ psi_tmp

    E_opt = pt.abs(pt.vdot(psi_tmp, psi)) ** 2

    N_E_opt_bell = E_opt * 2 ** L
    N_E_opt_ent = E_opt * 4 ** L

    Q_opt_ent = np.emath.logn(4, N_E_opt_ent)   # entanglement
    Q_opt_bell = np.emath.logn(2, N_E_opt_bell)  # Bell correlations

    return np.around(Q_opt_ent, 2), np.around(Q_opt_bell, 2)


# ---------------------------------------------------------------------------
# Two-qubit gates (CNOT / CZ) on the full Hilbert space
# ---------------------------------------------------------------------------

def get_CNOT(i: int, j: int, Id: pt.Tensor, X: dict, Z: dict) -> pt.Tensor:
    """
    Construct CNOT_{i -> j} as exp(i*pi*(Id - Z_i)(Id - X_j)/4)

    Args:
        i: control qubit index (1-based).
        j: target qubit index (1-based).
        Id: full identity operator on 2^L.
        X, Z: dictionaries of string operators (1..L).
    """
    if i == j:
        return Id
    return expm(1j * pt.pi * (Id - Z[i]) @ (Id - X[j]) * 0.25)


def get_CZ(i: int, j: int, Hadamard: dict, Id: pt.Tensor, X: dict, Z: dict) -> pt.Tensor:
    """
    Construct CZ_{i,j} via Hadamard on target followed by CNOT and Hadamard again.
    """
    cnot = get_CNOT(i, j, Id, X, Z)
    return Hadamard[j] @ cnot @ Hadamard[j]


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Tree parameters
    r = 1  # branching factor
    h = 9  # height

    graph = nx.balanced_tree(r, h)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    nx.draw(graph, with_labels=True, node_color="lightblue", ax=ax)

    L = len(graph.nodes)
    N = L

    print(f"L = {L}")
    print(h ** 2 + h + 1)

    # Build global operators
    D = 2 ** L
    Id = get_string_operator(id_local, L, 1)

    X = {}
    Y = {}
    Z = {}
    Hadamard = {}

    for i in range(1, L + 1):
        print(f"Building operators for site {i}")
        X[i] = get_string_operator(sigma_x, L, i)
        Y[i] = get_string_operator(sigma_y, L, i)
        Z[i] = get_string_operator(sigma_z, L, i)
        Hadamard[i] = get_string_operator(hadamard, L, i)

    # Basis in Z
    basis = generate_basis_via_sigma_z(L, Z)

    # Initial state |+...+>
    psi_ini = pt.ones((D,), dtype=pt.cdouble)
    psi_ini = psi_ini / pt.sqrt(pt.vdot(psi_ini, psi_ini))

    # Apply CZ gates along graph edges
    psi_graph = psi_ini.clone()
    for node_u, node_v in graph.edges:
        node_i = node_u + 1  # 1-based
        node_j = node_v + 1

 
        if node_i > N:
            node_i = N
        if node_j > N:
            node_j = N

        psi_graph = get_CZ(node_i, node_j, Hadamard, Id, X, Z) @ psi_graph

    psi = psi_graph

    # Measurement directions: restrict to {'Z', 'X'} as in original code
    sigma_plus_vec = ["Z", "X"]
    all_sigma_plus_sets = list(itertools.product(sigma_plus_vec, repeat=L))

    data_optimized_correlator = []

    for sigma_plus_set in all_sigma_plus_sets:
        Q_opt_ent, Q_opt_bell = get_correlator(psi, sigma_plus_set, L, X, Y, Z)

        data_dict_local = {
            "L": L,
            "Q_ent": np.around(Q_opt_ent, 2),
            "Q_bell": np.around(Q_opt_bell, 2),
            "sigma_plus_set": sigma_plus_set,
        }
        data_optimized_correlator.append(data_dict_local)

        string = f" L = {L} "
        string += "".join(sigma_plus_set) + "\n"
        string += f" | Q_ent = {Q_opt_ent:2.5f}"
        string += f" | Q_bell = {Q_opt_bell:2.5f}"
        print(string)

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
            rf"$ | Q_{{ent}} = $ {Q_ent} | $Q_{{bell}} = $ {Q_bell}" + "\n"
        )

    print(string_optimal_directions)

    title_string = f"L = {L} | r = {r} | h = {h}"
    title_string += f" | # = {len(data_Q_ent_max)}\n"
    title_string += string_optimal_directions

    ax.set_title(title_string, fontsize=16)
    plt.show()
