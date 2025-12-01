#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 11:29:43 2025

@author: Marcin Plodzien

PHYSICS CONTEXT:
----------------
Calcualtion of many-body Bell correlator for mixed graph states
1. **System**: N qubits in a Graph State (entangled via CZ gates).
2. **Noise**: Modeled via Kraus Operators (Quantum Channels).
3. **Metric**: Q correlator optimized over measurement angles; Q > 0 implies quantum entanglement. Q = N-2 is maximum for GHZ-like state.
"""

import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


import jax
import jax.numpy as jnp
import jax.random as jrand
import optax

# ==============================================================================
# 1. QUANTUM PRIMITIVES  
# ==============================================================================

# We use complex64 (float32 real/imag parts) which is standard for GPU/TPU acceleration.
# For high-precision physics (1e-12 accuracy), change to complex128.
complex_dtype = jnp.complex64

# --- Pauli Matrices ---
Id = jnp.eye(2, dtype=complex_dtype)
X  = jnp.array([[0., 1.], [1., 0.]], dtype=complex_dtype)       # Bit Flip (|0> <-> |1>)
Y  = 1j * jnp.array([[0., -1.], [1., 0.]], dtype=complex_dtype) # Bit + Phase Flip
Z  = jnp.array([[1., 0.], [0., -1.]], dtype=complex_dtype)      # Phase Flip (|1> -> -|1>)

# --- Initial State ---
# We start in the superposition state |+> = (|0> + |1>) / sqrt(2).
# The density matrix is rho = |+><+|.
rho0 = 0.5 * jnp.array([[1., 1.], [1., 1.]], dtype=complex_dtype) 

# --- CZ Gate (Control-Z) ---
# Diagonal form: diag(1, 1, 1, -1).
# We reshape this 4x4 matrix into a 4-leg tensor (2,2,2,2).
# This allows us to connect specific "legs" (indices) in a Tensor Network.
CZ_mat = jnp.diag(jnp.array([1., 1., 1., -1.], dtype=complex_dtype)).reshape(2,2,2,2)
CZ_flat = jnp.diag(jnp.array([1., 1., 1., -1.], dtype=complex_dtype))

def R_y_jax(angle):
    """
    Rotation around Y-axis: U = exp(-i * angle/2 * Y).
    Used to rotate the ground state |0> into |+>.
    """
    c = jnp.cos(angle / 2.0)
    s = jnp.sin(angle / 2.0)
    return jnp.array([[c, -s], [s, c]], dtype=complex_dtype)

def single_sigma_plus_jax(code: int):
    """
    Returns the measurement operator. 
    code: 0->X, 1->Y, 2->Z.
    IMPLEMENTATION: uses `jnp.stack` to create a lookup table tensor.
    """
    Sp_0 = 0.5 * (X + 1j * Y)
    Sp_1 = 0.5 * (Y + 1j * Z)
    Sp_2 = 0.5 * (Z + 1j * X)
    stack = jnp.stack([Sp_0, Sp_1, Sp_2])
    # .astype(int32) helps JAX know these are concrete indices
    return stack[code.astype(jnp.int32)]

def single_R_jax(theta_vec):
    """
    Arbitrary Rotation: U = exp(-i * theta . sigma).
    This is the variational ansatz.
    
    NOTE: We add `1e-12` to the norm. If theta is [0,0,0], the norm is 0.
    Calculating gradients involves (1/norm), which would explode (NaN) without this epsilon.
    """
    tz, ty, tx = theta_vec
    norm = jnp.sqrt(tx*tx + ty*ty + tz*tz) + 1e-12
    nx, ny, nz = tx/norm, ty/norm, tz/norm
    phi = 0.5 * jnp.pi * norm
    c, s = jnp.cos(phi), jnp.sin(phi)
    
    n_sigma = nz*Z + ny*Y + nx*X
    return c*Id - 1j*s*n_sigma

def build_local_O_jax(theta_vec, sigma_code: int):
    """
    Heisenberg Picture: O_rotated = U_dag * O * U.
    We optimize the measurement basis to align with the graph state.
    """
    R = single_R_jax(theta_vec)
    sigp = single_sigma_plus_jax(sigma_code)
    # R @ sigp @ R_dag
    return R @ sigp @ jnp.conj(R.T)

# --- NOISE CHANNELS (Open Quantum Systems) ---
def get_amp_damp_kraus(p):
    """
    Amplitude Damping Channel (Spontaneous Emission).
    K0: The system does not decay, but the amplitude of |1> decreases.
    K1: The system emits a photon and decays |1> -> |0>.
    Condition: K0^dag K0 + K1^dag K1 = I.
    """
    K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1-p)]], dtype=complex_dtype)
    K1 = jnp.array([[0.0, jnp.sqrt(p)], [0.0, 0.0]], dtype=complex_dtype)
    return [K0, K1]

# ==============================================================================
# 2. METHOD A: EXACT DENSITY MATRIX (DM)
# ==============================================================================
# OPTIMIZATION: "Tensor Reshaping"
# We treat the density matrix as a tensor with 2*N indices: N "bras" and N "kets".
# This allows us to apply a single-qubit gate to just ONE index without 
# building the full 2^N x 2^N identity matrix for the other qubits.

def dm_apply_1q(rho, U, site, L):
    """Applies single-qubit gate U via tensor contraction."""
    # 1. View rho as a tensor: (2, 2, ..., 2) [L times] for rows, same for cols
    rho_tens = rho.reshape((2,)*L + (2**L,))
    
    # 2. Contract U with the specific 'site' index (Left Mult: U * rho)
    rho_tens = jnp.tensordot(U, rho_tens, axes=[[1], [site]])
    
    # 3. Restore axis order (tensordot moves the new axis to front)
    rho_tens = jnp.moveaxis(rho_tens, 0, site)
    
    # 4. Right Mult: rho * U_dag. 
    # Trick: (A B)^T = B^T A^T. Transpose, Left Mult, Transpose back.
    rho_T = jnp.transpose(rho_tens.reshape(2**L, 2**L))
    rho_T_tens = rho_T.reshape((2,)*L + (2**L,))
    rho_T_tens = jnp.tensordot(jnp.conj(U), rho_T_tens, axes=[[1], [site]])
    rho_T_tens = jnp.moveaxis(rho_T_tens, 0, site)
    
    return jnp.transpose(rho_T_tens.reshape(2**L, 2**L))

def dm_apply_2q(rho, U, i, j, L):
    """Applies 2-qubit gate U via tensor contraction."""
    if i == j: return rho
    if i > j: i, j = j, i
    
    rho_tens = rho.reshape((2,)*L + (2**L,))
    
    # Move the two active qubits (i, j) to the front (indices 0, 1)
    rho_tens = jnp.moveaxis(rho_tens, (i, j), (0, 1)) 
    
    # Contract with 4x4 gate U
    rho_flat = U @ rho_tens.reshape((4, -1))
    
    # Restore shape and move axes back
    rho_tens = jnp.moveaxis(rho_flat.reshape((2,2)+(2,)*(L-2)+(2**L,)), (0,1), (i,j))
    
    # Repeat for Bra side
    rho_T = jnp.transpose(rho_tens.reshape(2**L, 2**L))
    rho_T_tens = rho_T.reshape((2,)*L + (2**L,))
    rho_T_tens = jnp.moveaxis(rho_T_tens, (i, j), (0, 1))
    rho_T_flat = jnp.conj(U) @ rho_T_tens.reshape((4, -1))
    rho_T_tens = jnp.moveaxis(rho_T_flat.reshape((2,2)+(2,)*(L-2)+(2**L,)), (0,1), (i,j))
    
    return jnp.transpose(rho_T_tens.reshape(2**L, 2**L))

def get_dm_kraus(noise_type, p_noise):
    if p_noise <= 0.0: return [Id]
    if noise_type == "depolarizing":
        # Simulates "white noise" on the Bloch sphere
        return [jnp.sqrt(1-p_noise)*Id, jnp.sqrt(p_noise/3)*X, jnp.sqrt(p_noise/3)*Y, jnp.sqrt(p_noise/3)*Z]
    elif noise_type == "bit_flip": 
        return [jnp.sqrt(1-p_noise)*Id, jnp.sqrt(p_noise)*X]
    elif noise_type == "phase_flip" or noise_type == "phase_damping": 
        return [jnp.sqrt(1-p_noise)*Id, jnp.sqrt(p_noise)*Z]
    elif noise_type == "amplitude_damping":
        return get_amp_damp_kraus(p_noise)
    return [Id]

def make_dm_loss(L, edges, sigma_codes, noise_type, p_noise):
    """Factory that creates the 'loss_fn' to be compiled by JAX."""
    Ry = R_y_jax(jnp.pi/2.0)
    kraus = get_dm_kraus(noise_type, p_noise)
    
    def get_rho():
        rho = jnp.zeros((2**L, 2**L), dtype=complex_dtype).at[0,0].set(1.0)
        # Init State
        for q in range(L): rho = dm_apply_1q(rho, Ry, q, L)
        # Gates + Noise
        for e in range(edges.shape[0]):
            u, v = int(edges[e,0]), int(edges[e,1])
            rho = dm_apply_2q(rho, CZ_flat, u, v, L)
            
            # Apply Noise Channel: Sum(K_i * rho * K_i_dag)
            rho_new = jnp.zeros_like(rho)
            for k in kraus: rho_new += dm_apply_1q(rho, k, u, L)
            rho = rho_new
            
            rho_new = jnp.zeros_like(rho)
            for k in kraus: rho_new += dm_apply_1q(rho, k, v, L)
            rho = rho_new
        return rho

    def loss_fn(theta):
        rho = get_rho()
        # Create global observable C_mat = O_1 (x) O_2 ...
        C_mat = build_local_O_jax(theta[0], sigma_codes[0])
        for i in range(1, L):
            C_mat = jnp.kron(C_mat, build_local_O_jax(theta[i], sigma_codes[i]))
        
        # We maximize the correlator <C>, so we minimize -<C>^2
        return -jnp.abs(jnp.trace(rho @ C_mat))**2
        
    return jax.jit(jax.value_and_grad(loss_fn))


# ==============================================================================
# 3. METHOD B: TENSOR NETWORK (TN)
# ==============================================================================
# Physics: Instead of a matrix, we view the channel as a Superoperator Tensor.
# We let `jnp.einsum` handle the contraction path.
# This is efficient for low entanglement (Line graphs) but slow for high entanglement.

def get_tn_superop(noise_type, p_noise):
    """Constructs the 4-leg Superoperator Tensor for the noise channel."""
    if p_noise <= 0.0: K_ops = [Id]
    elif noise_type == "depolarizing":
        K_ops = [jnp.sqrt(1-p_noise)*Id, jnp.sqrt(p_noise/3)*X, jnp.sqrt(p_noise/3)*Y, jnp.sqrt(p_noise/3)*Z]
    elif noise_type == "bit_flip": 
        K_ops = [jnp.sqrt(1-p_noise)*Id, jnp.sqrt(p_noise)*X]
    elif noise_type == "phase_flip" or noise_type == "phase_damping": 
        K_ops = [jnp.sqrt(1-p_noise)*Id, jnp.sqrt(p_noise)*Z]
    elif noise_type == "amplitude_damping":
        K_ops = get_amp_damp_kraus(p_noise)
    else: K_ops = [Id]
    
    K_stack = jnp.stack(K_ops)
    # Contraction: T_{acbd} = sum_i K_{ab}^i * conj(K_{cd}^i)
    return jnp.einsum('iab, icd -> acbd', K_stack, jnp.conj(K_stack))

def build_tn_contraction(theta, L, edges, sigma_codes, noise_tensor):
    """Builds lists of tensors and indices for einsum."""
    tensors, indices = [], []
    idx_counter = 0
    open_legs = [] 
    
    # 1. State Prep
    for i in range(L):
        k, b = idx_counter, idx_counter+1; idx_counter += 2
        open_legs.append([k, b])
        tensors.append(rho0); indices.append([k, b])
    
    # 2. Gates + Noise
    for u, v in edges:
        ku, bu = open_legs[u]
        kv, bv = open_legs[v]
        ku_cz, kv_cz = idx_counter, idx_counter+1; idx_counter += 2
        bu_cz, bv_cz = idx_counter, idx_counter+1; idx_counter += 2
        
        tensors.append(CZ_mat); indices.append([ku_cz, kv_cz, ku, kv])
        tensors.append(CZ_mat); indices.append([bu_cz, bv_cz, bu, bv])
        
        ku_noise, bu_noise = idx_counter, idx_counter+1; idx_counter += 2
        tensors.append(noise_tensor); indices.append([ku_noise, bu_noise, ku_cz, bu_cz])
        
        kv_noise, bv_noise = idx_counter, idx_counter+1; idx_counter += 2
        tensors.append(noise_tensor); indices.append([kv_noise, bv_noise, kv_cz, bv_cz])
        
        open_legs[u] = [ku_noise, bu_noise]
        open_legs[v] = [kv_noise, bv_noise]

    # 3. Measure
    for i in range(L):
        b, k = open_legs[i]
        O = build_local_O_jax(theta[i], sigma_codes[i])
        tensors.append(O); indices.append([b, k])

    return tensors, indices

def make_tn_loss(L, edges, sigma_codes, noise_type, p_noise):
    noise_T = get_tn_superop(noise_type, p_noise)
    def loss_fn(theta):
        ts, inds = build_tn_contraction(theta, L, edges, sigma_codes, noise_T)
        args = [x for pair in zip(ts, inds) for x in pair]
        # 'optimize=greedy' finds the best contraction order
        res = jnp.einsum(*args, optimize='greedy')
        return -jnp.abs(res)**2
    return jax.jit(jax.value_and_grad(loss_fn))


# ==============================================================================
# 4. METHOD C: MONTE CARLO (MC) - The "Unravelling" Way
# ==============================================================================
# Instead of evolving the density matrix (ensemble), we evolve a single 
# pure state |psi> and introduce stochastic "jumps".
# We average over Ntraj trajectories.

def mc_apply_gate(psi, U, i, L):
    """Applies gate to pure state vector."""
    psi = psi.reshape((2,)*L)
    psi = jnp.tensordot(U, psi, axes=[[1], [i]])
    return jnp.moveaxis(psi, 0, i).reshape(-1)

def mc_apply_2q(psi, U, i, j, L):
    if i > j: i, j = j, i
    psi = psi.reshape((2,)*L)
    psi = jnp.moveaxis(psi, (i, j), (0, 1))
    psi = (U @ psi.reshape(4, -1)).reshape((2,2)+(2,)*(L-2))
    return jnp.moveaxis(psi, (0, 1), (i, j)).reshape(-1)

def make_mc_noise(noise_type, p_noise):
    """
    Returns a FUNCTION `noise(psi, key)` that performs a stochastic jump.
    
    JAX IMPLEMENTATION DETAIL:
    JAX cannot handle Python's `if random() < p`. 
    We must use `jax.lax.cond` or `jax.lax.select`.
    This compiles the "if" statement into the computation graph.
    """
    if p_noise <= 0.0: 
        return lambda psi, k, s, l: (psi, k)
    
    if noise_type == "bit_flip":
        def noise(psi, key, site, L):
            b = jrand.bernoulli(key, p_noise)
            # Efficient branchless selection
            return mc_apply_gate(psi, jax.lax.select(b, X, Id), site, L), key
        return noise
    
    if noise_type == "depolarizing":
        def noise(psi, key, site, L):
            r = jrand.uniform(key)
            # Thresholds
            th1, th2, th3 = 1-p_noise, 1-2*p_noise/3, 1-p_noise/3
            # Nested conditions (like a switch case)
            def dI(_): return mc_apply_gate(psi, Id, site, L)
            def dX(_): return mc_apply_gate(psi, X, site, L)
            def dY(_): return mc_apply_gate(psi, Y, site, L)
            def dZ(_): return mc_apply_gate(psi, Z, site, L)
            return jax.lax.cond(r<th1, dI, lambda _: jax.lax.cond(r<th2, dX, lambda _: jax.lax.cond(r<th3, dY, dZ, None), None), None), key
        return noise
    
    if noise_type == "phase_flip" or noise_type == "phase_damping":
        def noise(psi, key, site, L):
            b = jrand.bernoulli(key, p_noise)
            return mc_apply_gate(psi, jax.lax.select(b, Z, Id), site, L), key
        return noise
    
    # Non-Unitary Jump (Amplitude Damping)
    if noise_type == "amplitude_damping":
        K0, K1 = get_amp_damp_kraus(p_noise)
        def noise(psi, key, site, L):
            # Calculate both branches
            phi_0 = mc_apply_gate(psi, K0, site, L) # No Jump
            phi_1 = mc_apply_gate(psi, K1, site, L) # Jump
            
            # Probability of jump is the norm of the 'jump' branch
            p_jump = jnp.real(jnp.vdot(phi_1, phi_1))
            
            r = jrand.uniform(key)
            
            # We NORMALIZE the state after the decision
            def do_jump(_): return phi_1 / (jnp.sqrt(p_jump) + 1e-12)
            def no_jump(_): return phi_0 / (jnp.sqrt(1.0 - p_jump) + 1e-12)
            
            return jax.lax.cond(r < p_jump, do_jump, no_jump, None), key
        return noise
        
    return lambda psi, k, s, l: (psi, k)

def make_mc_funcs(L, edges, sigma_codes, noise_type, p_noise, Ntraj):
    noise_fn = make_mc_noise(noise_type, p_noise)
    Ry = R_y_jax(jnp.pi/2.0)
    
    def run_traj(theta, key):
        """Simulates ONE trajectory."""
        psi = jnp.zeros(2**L, dtype=complex_dtype).at[0].set(1.0)
        for q in range(L): psi = mc_apply_gate(psi, Ry, q, L)
        k_loc = key
        for u, v in edges:
            psi = mc_apply_2q(psi, CZ_flat, int(u), int(v), L)
            k_loc, ku, kv = jrand.split(k_loc, 3)
            # Random Noise Application
            psi, _ = noise_fn(psi, ku, int(u), L)
            psi, _ = noise_fn(psi, kv, int(v), L)
        
        # Numerical Safety Normalize
        psi /= jnp.linalg.norm(psi) + 1e-12
        
        # Measure Overlap
        phi = psi
        for i in range(L):
            O = build_local_O_jax(theta[i], sigma_codes[i])
            phi = mc_apply_gate(phi, O, i, L)
        return jnp.vdot(psi, phi)

    def stats_fn(theta, key):
        """
        Runs Ntraj trajectories in PARALLEL using `jax.vmap`.
        This is much faster than a Python for loop.
        """
        keys = jrand.split(key, Ntraj)
        vals = jax.vmap(lambda k: run_traj(theta, k))(keys)
        mean, std = jnp.mean(vals), jnp.std(vals)
        return mean, std/jnp.sqrt(Ntraj) # Return Mean and SEM
    
    def loss_fn(theta, key):
        mean, _ = stats_fn(theta, key)
        return -jnp.abs(mean)**2
        
    return jax.jit(stats_fn), jax.jit(jax.value_and_grad(loss_fn, argnums=0))

# ==============================================================================
# 5. EXECUTION LOOP
# ==============================================================================

def run_opt(theta_init, key, loss_grad_fn, steps=200, is_mc=False):
    """
    Runs the Adam optimizer loop.
    
    JAX OPTIMIZATION:
    We use `jax.lax.scan` instead of a Python `for` loop.
    `scan` compiles the entire loop into a single XLA kernel, reducing overhead.
    """
    opt = optax.adam(0.02)
    st = opt.init(theta_init)
    
    @jax.jit
    def step(carry, _):
        th, s, k = carry
        k, sk = jrand.split(k)
        args = (th, sk) if is_mc else (th,)
        
        l, g = loss_grad_fn(*args)
        
        up, s = opt.update(g, s, th)
        th = optax.apply_updates(th, up)
        return (th, s, k), l
        
    (th_f, _, _), ls = jax.lax.scan(step, (theta_init, st, key), None, length=steps)
    return th_f, -ls[-1] # Return best loss (converted to positive)

def plot_graph_topology(G, title, fname):
    """
    Saves a visualization of the graph topology.
    This helps verifying the structure (Grid vs Star vs Turan).
    """
    plt.figure(figsize=(6, 6))
    # Use different layouts for different graph types roughly
    if "Grid" in title:
        pos = nx.kamada_kawai_layout(G) # Often good for grids
    else:
        pos = nx.spring_layout(G, seed=42)
    
    nx.draw(G, pos, 
            with_labels=True, 
            node_color='skyblue', 
            edge_color='gray', 
            node_size=500,
            font_weight='bold')
    plt.title(title)
    plt.savefig(fname, dpi=150)
    plt.close()

def make_graph(kind, N, er_p=None, er_m=None, grid_shape=None, turan_r=None, seed=42):
    """
    Topology factory.
    
    New Features:
    - kind="grid": Creates a 2D Lattice.
      Uses 'grid_shape=(rows, cols)'. If None, auto-calculates best square.
    - kind="turan": Creates a Turan Graph T(N, r).
      Uses 'turan_r' (number of partitions).
    """
    if kind == "star": G = nx.star_graph(N-1)
    elif kind == "line": G = nx.path_graph(N)
    elif kind == "cycle": G = nx.cycle_graph(N)
    elif kind == "complete": G = nx.complete_graph(N)
    elif kind == "wheel": G = nx.wheel_graph(N)
    
    elif kind == "grid":
        if grid_shape is not None:
            m, n = grid_shape
            if m * n != N:
                raise ValueError(f"Grid shape {m}x{n}={m*n} does not match N={N}")
        else:
            # Auto-calculate factors closest to sqrt(N)
            m = int(np.sqrt(N))
            while N % m != 0:
                m -= 1
            n = N // m
        G = nx.grid_2d_graph(m, n)
        
    elif kind == "turan":
        if turan_r is None:
            raise ValueError("Must provide 'turan_r' parameter for Turan graph")
        G = nx.turan_graph(N, turan_r)
        
    elif kind == "erdos_renyi_gnp": G = nx.erdos_renyi_graph(N, er_p, seed=seed)
    elif kind == "erdos_renyi_gnm": G = nx.gnm_random_graph(N, er_m, seed=seed)
    else: raise ValueError(f"Unknown graph: {kind}")
    
    # Standardize node labels to 0..N-1 for JAX
    G = nx.convert_node_labels_to_integers(G)
    
    return G, np.array(G.edges, dtype=np.int32)

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # Toggle methods here
    ENABLE_DM = True
    ENABLE_TN = True
    ENABLE_MC = True
    
    # -------------------------------------------------------------
    # AVAILABLE GRAPH TYPES:
    # 1. "star"      (N-1 outer nodes connected to 1 center)
    # 2. "line"      (1D chain of qubits)
    # 3. "cycle"     (Ring of qubits)
    # 4. "complete"  (All-to-all connectivity - High Entanglement!)
    # 5. "grid"      (2D Lattice. Set 'grid_shape=(rows,cols)')
    # 6. "turan"     (Multipartite. Set 'turan_r' partitions)
    # 7. "erdos_renyi_gnp" (Random graph, probability p)
    # -------------------------------------------------------------
    graph_type = "star"
    N = 12
    
    # --- GRAPH PARAMETERS ---
    # Only used if graph_type matches
    grid_shape = (3, 3)  # For 'grid': Must multiply to N
    turan_r = 3          # For 'turan': Partitions
    er_p = 0.4           # For random graph
    
    noise_type_vec = ["depolarizing",    "phase_damping", "amplitude_damping"]
    # noise_type_vec = ["depolarizing",  "phase_damping"]
    # noise_type_vec = ["phase_flip", "phase_damping"]  
    noise_type_vec = ["amplitude_damping"]
    # noise_type_vec = ["bit_flip"]
    
    
 
    noise_type_vec = ["depolarizing",    "bit_flip", "amplitude_damping"]

    # noise_type_vec = ["amplitude_damping"]
    p_noise_values = np.linspace(0.0, 0.25, 10) 
    
    DM_MAX_N = 10 # DM uses O(4^N) RAM.
    Ntraj = 400 # Number of trajectories for MC
    steps = 200  # Optimization steps
    
    for noise_type in noise_type_vec:
        er_m, seed = 10, 123
        should_run_dm = ENABLE_DM and (N <= DM_MAX_N)
        
        # --- GENERATE GRAPH & STRINGS ---
        G, edges_np = make_graph(graph_type, N, er_p=er_p, er_m=er_m, 
                                 grid_shape=grid_shape, turan_r=turan_r, seed=seed)
        L = len(G.nodes)
        
        # Construct a descriptive string for filenames/titles
        graph_details_str = f"N{N}_{graph_type}"
        if graph_type == "grid":
            if grid_shape is None: 
                # Re-calculate implied shape
                m = int(np.sqrt(N))
                while N % m != 0: m -= 1
                n = N // m
                graph_details_str += f"_{m}x{n}"
            else:
                graph_details_str += f"_{grid_shape[0]}x{grid_shape[1]}"
        elif graph_type == "turan":
            graph_details_str += f"_r{turan_r}"
        
        
 

        print(f"\n=== {graph_details_str} | {noise_type} ===")
        
        # Save Topology Plot
        os.makedirs("results", exist_ok=True)
        topo_fname = f"results/topo_{graph_details_str}.png"
        plot_graph_topology(G, f"Topology: {graph_details_str}", topo_fname)
        
        sigma_codes = jnp.array([1]*L, dtype=jnp.int32)
        data_records = []
        
        key = jrand.PRNGKey(42)
        theta_init = 0.1 * jrand.normal(key, (L, 3))
        
        # --- FILENAME LOGIC ---
        active_methods = []
        if should_run_dm: active_methods.append("DM")
        if ENABLE_TN: active_methods.append("TN")
        if ENABLE_MC: active_methods.append("MC")
        methods_str = "-".join(active_methods)
        
        # --- MONITOR HEADER ---
        header_fmt = "{:<8} | {:<15} | {:<15} | {:<25}"
        print("-" * 75)
        print(header_fmt.format("p_noise", "DM (Q | Time)", "TN (Q | Time)", "MC (Q | Err | Time)"))
        print("-" * 75)
        
        for p_noise in p_noise_values:
            
            # 1. DENSITY MATRIX
            vals_dm = {'Q': np.nan, 'Time': np.nan, 'Angles': None}
            if should_run_dm:
                t0 = time.time()
                loss_grad_dm = make_dm_loss(L, edges_np, sigma_codes, noise_type, p_noise)
                th_dm, raw_dm = run_opt(theta_init, key, loss_grad_dm, steps)
                dt_dm = time.time() - t0
                Q_dm = float(jnp.log2(max(raw_dm, 1e-15) * 2**L))
                vals_dm = {'Q': Q_dm, 'Time': dt_dm, 'Angles': np.array(th_dm).tolist()}
                str_dm = f"{Q_dm:.4f} {dt_dm:.2f}s"
            else:
                str_dm = "SKIP"
            
            # 2. TENSOR NETWORK
            vals_tn = {'Q': np.nan, 'Time': np.nan, 'Angles': None}
            if ENABLE_TN:
                t0 = time.time()
                loss_grad_tn = make_tn_loss(L, edges_np, sigma_codes, noise_type, p_noise)
                th_tn, raw_tn = run_opt(theta_init, key, loss_grad_tn, steps)
                dt_tn = time.time() - t0
                Q_tn = float(jnp.log2(max(raw_tn, 1e-15) * 2**L))
                vals_tn = {'Q': Q_tn, 'Time': dt_tn, 'Angles': np.array(th_tn).tolist()}
                str_tn = f"{Q_tn:.4f} {dt_tn:.2f}s"
            else:
                str_tn = "OFF"
            
            # 3. MONTE CARLO
            vals_mc = {'Q': np.nan, 'Time': np.nan, 'Err': np.nan, 'Angles': None}
            if ENABLE_MC:
                t0 = time.time()
                stats_mc, loss_grad_mc = make_mc_funcs(L, edges_np, sigma_codes, noise_type, p_noise, Ntraj)
                th_mc, _ = run_opt(theta_init, key, loss_grad_mc, steps, is_mc=True)
                
                # Eval stats
                key, k_eval = jrand.split(key)
                mean, sem = stats_mc(th_mc, k_eval)
                raw_mc = float(jnp.abs(mean)**2)
                raw_err = float(2 * jnp.abs(mean) * sem) 
                
                Q_mc = float(jnp.log2(max(raw_mc, 1e-15) * 2**L))
                Q_err = (1.0/(max(raw_mc, 1e-15)*np.log(2))) * raw_err
                
                dt_mc = time.time() - t0
                vals_mc = {'Q': Q_mc, 'Err': Q_err, 'Time': dt_mc, 'Angles': np.array(th_mc).tolist()}
                str_mc = f"{Q_mc:.4f} +/-{Q_err:.4f} {dt_mc:.2f}s"
            else:
                str_mc = "OFF"
            
            print(header_fmt.format(f"{p_noise:.4f}", str_dm, str_tn, str_mc))
            
            record = {
                "N": N, "graph_type": graph_type, "noise_type": noise_type, "p_noise": p_noise,
                "methods_ran": methods_str,
                # Graph Params saved to record
                "grid_shape": grid_shape if graph_type == "grid" else None,
                "turan_r": turan_r if graph_type == "turan" else None,
                "edges": edges_np.tolist(),
                # Results
                "dm_Q": vals_dm['Q'], "dm_Time": vals_dm['Time'], "dm_angles": vals_dm['Angles'],
                "tn_Q": vals_tn['Q'], "tn_Time": vals_tn['Time'], "tn_angles": vals_tn['Angles'],
                "mc_Q": vals_mc['Q'], "mc_Q_err": vals_mc['Err'], "mc_Time": vals_mc['Time'], "mc_angles": vals_mc['Angles']
            }
            data_records.append(record)
            
        pkl_fname = f"results/data_{graph_details_str}_{noise_type}_methods-{methods_str}.pkl"
        pd.DataFrame(data_records).to_pickle(pkl_fname)
        print(f"[Save] Data saved to {pkl_fname}")