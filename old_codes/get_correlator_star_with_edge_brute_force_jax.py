#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marcin Plodzien
"""


import itertools

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax


# -----------------------------------------------------------------------------
# Local single-qubit operators (2x2 only)
# -----------------------------------------------------------------------------

id_local = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128)
sigma_x = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
sigma_y = 1j * jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.complex128)
sigma_z = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)

hadamard = (
    1.0 / jnp.sqrt(jnp.array(2.0))
    * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128)
)


# -----------------------------------------------------------------------------
# CZ gates acting directly on state vector  
# -----------------------------------------------------------------------------

@jit
def apply_cz_edges(psi: jnp.ndarray, edges: jnp.ndarray, L: int) -> jnp.ndarray:
    """
    Apply CZ gates along all edges to state vector psi.

    psi:   shape (2**L,)
    edges: shape (num_edges, 2) with 1-based qubit indices (q1, q2)
    L:     number of qubits

    CZ_{q1,q2} acts as a phase -1 if both qubits q1 and q2 are |1>, else 1.
    """
    D = psi.shape[0]
    idxs = jnp.arange(D, dtype=jnp.int32)  # computational basis indices

    def body(psi_in, edge):
        q1, q2 = edge  # 1-based indices
        shift1 = L - q1
        shift2 = L - q2

        mask1 = jnp.left_shift(jnp.int32(1), shift1)
        mask2 = jnp.left_shift(jnp.int32(1), shift2)

        cond = ((idxs & mask1) != 0) & ((idxs & mask2) != 0)
        phase = jnp.where(cond, -1.0 + 0j, 1.0 + 0j)

        psi_out = psi_in * phase
        return psi_out, None

    psi_out, _ = lax.scan(body, psi, edges)
    return psi_out


# -----------------------------------------------------------------------------
# Per-site branches for applying a 2x2 operator (JIT-safe)
# -----------------------------------------------------------------------------

def make_site_branch(L: int, site: int):
    """
    Create a branch function that applies a 2x2 operator `op`
    on qubit `site` (0-based from left) of a length-2**L state `psi`.

    This branch has only *static* permutations, so it's JIT-friendly.
    """

    # Static permutations, computed once in Python
    perm = (site,) + tuple(i for i in range(L) if i != site)
    inv_perm = tuple(sorted(range(L), key=lambda k: perm[k]))

    def branch(psi_and_op):
        psi, op = psi_and_op  # psi: (2**L,), op: (2,2)

        psi_tensor = psi.reshape((2,) * L)          # (2,2,...,2)
        psi_perm = jnp.transpose(psi_tensor, perm)  # bring `site` to axis 0

        shape_perm = psi_perm.shape
        psi_flat = psi_perm.reshape(2, -1)          # (2, 2**(L-1))

        psi_new_flat = op @ psi_flat                # apply 2x2 op
        psi_new_perm = psi_new_flat.reshape(shape_perm)

        psi_new = jnp.transpose(psi_new_perm, inv_perm).reshape(-1)
        return psi_new

    return branch


# -----------------------------------------------------------------------------
# Correlators (entanglement / Bell), JIT + VMAP
# -----------------------------------------------------------------------------

def make_correlator_fn(L: int):
    """
    Build a JIT-compiled correlator function that evaluates all
    sigma_plus patterns in parallel via vmap.

    Uses local 2x2 Pauli matrices + per-site branches.
    """

    # One branch per site for local op application
    site_branches = [make_site_branch(L, s) for s in range(L)]

    @jit
    def correlators_for_patterns(
        psi: jnp.ndarray,
        patterns: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        psi:      shape (2**L,)
        patterns: shape (num_patterns, L), integer codes:
                  0 -> "Z", 1 -> "X", 2 -> "Y"

        Returns:
            Q_ent_all:  shape (num_patterns,)
            Q_bell_all: shape (num_patterns,)
        """

        def sigma_plus_from_axis(axis_code: jnp.ndarray) -> jnp.ndarray:
            # axis_code ∈ {0,1,2}  (Z, X, Y)
            return lax.switch(
                axis_code,
                [
                    # Z: σ⁺ = 0.5 (X + iY)
                    lambda _: 0.5 * (sigma_x + 1j * sigma_y),
                    # X: σ⁺ = 0.5 (Y + iZ)
                    lambda _: 0.5 * (sigma_y + 1j * sigma_z),
                    # Y: σ⁺ = 0.5 (Z + iX)
                    lambda _: 0.5 * (sigma_z + 1j * sigma_x),
                ],
                operand=None,
            )

        def single_pattern_correlator(pattern: jnp.ndarray):
            # pattern: shape (L,), entries in {0,1,2}
            psi_tmp = psi

            def body(k, psi_in):
                # Apply from rightmost qubit (L-1) down to 0
                site = L - 1 - k          # traced scalar
                axis = pattern[site]      # traced scalar in {0,1,2}

                sigma_plus_i = sigma_plus_from_axis(axis)

                # Choose branch by `site`, but each branch has static permutations.
                psi_out = lax.switch(site, site_branches, (psi_in, sigma_plus_i))
                return psi_out

            psi_final = lax.fori_loop(0, L, body, psi_tmp)

            E_opt = jnp.abs(jnp.vdot(psi_final, psi)) ** 2

            N_E_opt_bell = E_opt * (2 ** L)
            N_E_opt_ent = E_opt * (4 ** L)

            Q_ent = jnp.log(N_E_opt_ent) / jnp.log(4.0)
            Q_bell = jnp.log(N_E_opt_bell) / jnp.log(2.0)

            return Q_ent, Q_bell

        Q_ent_all, Q_bell_all = vmap(single_pattern_correlator, in_axes=0)(patterns)
        return Q_ent_all, Q_bell_all

    return correlators_for_patterns


# -----------------------------------------------------------------------------
# Main simulation
# -----------------------------------------------------------------------------

def main():
    # Number of leaves in the star graph (center is node 0)
    N = 12 # => L = N + 1 = 9

    # Star graph with center node 0 and leaves 1..N
    graph = nx.star_graph(N)

    # System size: one qubit per node
    L = graph.number_of_nodes()
    print(f"L = {L}")

    D = 2 ** L

    # Initial state: |+>^L
    psi_ini = jnp.ones((D,), dtype=jnp.complex128)
    psi_ini = psi_ini / jnp.sqrt(jnp.vdot(psi_ini, psi_ini))


    axis_map = {"Z": 0, "X": 1, "Y": 2}
    sigma_plus_vec = ("Z", "X")  # allowed directions

    all_sigma_plus_sets = list(itertools.product(sigma_plus_vec, repeat=L))
    patterns_np = np.array(
        [[axis_map[c] for c in pattern] for pattern in all_sigma_plus_sets],
        dtype=np.int32,
    )
    patterns = jnp.array(patterns_np)  # (num_patterns, L)

    # Build JIT'ed correlator
    correlators_for_patterns = make_correlator_fn(L)

 
    for i in range(1, N + 1):
         if i + 1 <= N:
            j = i + 1
        else:
            j = 1
        graph.add_edge(i, j)

 
        edge_nodes = [node for node in graph.nodes if node != 0]
        pos = nx.circular_layout(graph.subgraph(edge_nodes))
        center_node = 0
        pos[center_node] = np.array([0.0, 0.0])

        # Apply CZ along all edges via JAX (phases only)
        # Map graph nodes (0..N) -> site indices (1..L)
        edges_sites = np.array(
            [[u + 1, v + 1] for (u, v) in graph.edges()],
            dtype=np.int32,
        )
        edges_jax = jnp.array(edges_sites)

        psi_graph = apply_cz_edges(psi_ini, edges_jax, L)
        psi = psi_graph  # many-body state after all CZs

        # Compute all correlators for all sigma_plus patterns (JIT + VMAP)
        Q_ent_all, Q_bell_all = correlators_for_patterns(psi, patterns)

 
        Q_ent_np = np.array(Q_ent_all)
        Q_bell_np = np.array(Q_bell_all)

        data_optimized_correlator = []
        for idx, pattern in enumerate(all_sigma_plus_sets):
            Q_ent = np.round(Q_ent_np[idx], 2)
            Q_bell = np.round(Q_bell_np[idx], 2)
            data_optimized_correlator.append(
                {
                    "L": L,
                    "Q_ent": Q_ent,
                    "Q_bell": Q_bell,
                    "sigma_plus_set": pattern,
                }
            )

            s = f" L = {L} "
            s += "".join(pattern) + "\n"
            s += f" | Q_ent = {Q_ent:2.5f}"
            s += f" | Q_bell = {Q_bell:2.5f}"
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
                rf"$ | Q_{{ent}} = $ {Q_ent} | $Q_{{bell}} = $ {Q_bell}" + "\n"
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
