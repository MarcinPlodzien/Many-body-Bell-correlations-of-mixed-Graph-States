# Many-body Bell Correlations in Mixed Graph States

## Overview

This repository implements a high-performance numerical framework to study **many-body Bell correlations** in quantum **Graph States** subject to noise channels.

The core physical problem addresses the robustness of multipartite entanglement in mixed states. We consider a graph state $\rho_G$ subjected to a quantum noise channel $\Lambda$. The objective is to maximize the expectation value $\mathcal{E}$ of a many-body Bell correlator. The optimization problem is defined as:

$$
{\cal E} = \max_{\vec{\theta}} \left| \mathrm{Tr}[\Lambda(\rho_G) \cdot \mathcal{B}(\vec{\theta})] \right|^2
$$

$$
Q = \log_2({\cal E}) + N
$$

Where:

* $\Lambda(\cdot)$ is the quantum noise channel (e.g., Amplitude Damping, Depolarizing) represented by Kraus operators $\{K_i\}$ such that $\Lambda(\rho) = \sum_i K_i \rho K_i^\dagger$.

* $\mathcal{B}(\vec{\theta})$ is the **Variational Bell Operator**, constructed as the tensor product of local $\sigma_{+}$ operators rotated into an optimal measurement basis:

$$
\mathcal{B}(\vec{\theta}) = \bigotimes_{j=1}^N U_j(\theta_j) \sigma^{+}_j U_j^\dagger(\theta_j)
$$

Here, $\sigma_{+}$ represents the local raising operator (e.g., $|0\rangle\langle 1|$ in the computational basis), and $U_j(\theta_j)$ is a parameterized local unitary rotation acting on qubit $j$.

To solve this optimization problem efficiently across different system sizes $N$, this framework implements and benchmarks three distinct numerical methods.

## Related Research

The **Bell correlator** has been defined and used in the following papers (among others):

1. **Entanglement in graph states and its detection**
   *Reports on Progress in Physics* (2024)
   [IOP Science](https://iopscience.iop.org/article/10.1088/1361-6633/adecc0) | [arXiv:2410.12487](https://arxiv.org/pdf/2410.12487)

2. **Variational quantum algorithms for graph state entanglement**
   *Physical Review A* **110**, 032428 (2024)
   [APS Journals](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.110.032428) | [arXiv:2406.10662](https://arxiv.org/pdf/2406.10662)

## Numerical Methodologies

This project implements three solvers for the simulation of noisy quantum evolution. The framework relies on **JAX** for Just-In-Time (JIT) compilation and Automatic Differentiation (AD) to optimize the variational parameters $\vec{\theta}$.

### 1. Exact Density Matrix (DM)

* **Formalism**:  Represents the state as a dense matrix of dimension $2^N \times 2^N$. Noise channels are applied via the Kraus Operator sum representation: $\rho' = \sum_k K_k \rho K_k^\dagger$.

* **Numerical Technique**: Utilizes a tensor reshaping strategy to avoid constructing full sparse matrices for local operations. The density matrix is treated as a tensor with shape `(2, 2, ..., 2)` for the bra-indices and `(2, 2, ..., 2)` for the ket-indices.

* **Computational Complexity**: Memory scales as $O(4^N)$. This method provides the exact reference solution but is strictly limited to small systems ($N \lesssim 12$).

### 2. Tensor Network (TN)

* **Formalism**: Represents the quantum channel $\Lambda$ and state evolution as a contraction of tensors.

* **Implementation**: The noise channel is vectorized into a rank-4 **Superoperator Tensor**. The expectation value $\mathcal{E}$ is computed by contracting the network of superoperators along the graph structure.

* **Computational Complexity**: Memory scaling is linear $O(N)$ for low-treewidth graphs (e.g., 1D chains), but contraction time scales exponentially with the entanglement entropy (treewidth) of the graph.

### 3. Monte Carlo Wavefunction (MC) / Quantum Trajectories

* **Formalism**: Stochastic unravelling of Quantum Channels (Discrete-time Quantum Trajectories).

* **Implementation**: Evolves a single pure state $|\psi\rangle$ of dimension $2^N$. Dissipative processes defined by $\Lambda$ are modeled as stochastic **Quantum Jumps** determined by Monte Carlo sampling.

* **Numerical Technique**: Leveraging JAX's `vmap`, thousands of independent trajectories are executed in parallel on GPU/TPU to estimate the ensemble average: $\rho \approx \frac{1}{M} \sum_{i=1}^M |\psi_i\rangle\langle\psi_i|$.

* **Computational Complexity**: Memory scales as $O(2^N)$. This allows for the simulation of significantly larger systems ($N \approx 20-30$) compared to the DM approach, subject to statistical sampling error $\propto 1/\sqrt{M}$.


## Simulation & Usage

The workflow is divided into simulation (generating data) and analysis (visualization).

### 1. Simulation (`01_get_bell_correlations.py`)

This script executes the variational optimization loop. It sweeps over the noise parameter $p$ in channel $\Lambda$, optimizes the measurement angles $\vec{\theta}$ using the Adam optimizer, and computes the expectation value $\mathcal{E}$ and Bell correlation $Q$.

```bash
python 01_get_bell_correlations.py
```

**Configuration:**
Modify the `if __name__ == "__main__":` block to define the physical system:

* `N`: System size (number of qubits).

* `graph_type`: Topology of the entangled state (`star`, `grid`, `turan`, etc.).

* `noise_type_vec`: List of quantum channels to investigate (e.g., `amplitude_damping`, `depolarizing`).

**Output:**
Data is serialized into Pandas DataFrames and saved as `.pkl` files in the `results/` directory, containing optimal angles, metrics ($\mathcal{E}, Q$), and topology data.

### 2. Visualization (`02_plot_results.py`)

This script generates figures comparing the results of the different solvers and noise models.

```bash
python 02_plot_results.py
```
