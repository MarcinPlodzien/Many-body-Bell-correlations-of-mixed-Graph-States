Benchmarking Bell Correlations in Mixed Graph States

A JAX-based computational framework for simulating Variational Quantum Algorithms (VQA) in open quantum systems.

Scientific Overview

This repository implements a high-performance numerical framework to study many-body Bell correlations (specifically the $Q$ correlator) in quantum Graph States subject to realistic environmental noise.

The core physical problem addresses the robustness of multipartite entanglement in mixed states. We consider a graph state $\rho_G$ subjected to a quantum noise channel $\Lambda$. The objective is to maximize the expectation value $\mathcal{E}$ of a many-body Bell correlator. The optimization problem is defined as:

$$\mathcal{E} = \max_{\vec{\theta}} \left| \mathrm{Tr}[\Lambda(\rho_G) \cdot \mathcal{B}(\vec{\theta})] \right|^2$$

$$Q = \log_2(\mathcal{E}) + N$$

Where:

$\Lambda(\cdot)$ is the quantum noise channel (e.g., Amplitude Damping, Depolarizing) represented by Kraus operators $\{K_i\}$ such that $\Lambda(\rho) = \sum_i K_i \rho K_i^\dagger$.

$\mathcal{B}(\vec{\theta})$ is the Variational Bell Operator, constructed as the tensor product of local $\sigma_{+}$ operators rotated into an optimal measurement basis:

$$\mathcal{B}(\vec{\theta}) = \bigotimes_{j=1}^N U_j(\theta_j) \sigma_{+} U_j^\dagger(\theta_j)$$

Here, $\sigma_{+}$ represents the local raising operator (e.g., $|0\rangle\langle 1|$ in the computational basis), and $U_j(\theta_j)$ is a parameterized local unitary rotation acting on qubit $j$.

To solve this optimization problem efficiently across different system sizes $N$, this framework implements and benchmarks three distinct numerical methods.

Related Research

The Bell correlator $Q$ maximized in this project and the variational protocols employed are derived directly from the theoretical framework and definitions established in the following papers:

Entanglement in graph states and its detection
Reports on Progress in Physics (2024)
IOP Science | arXiv:2410.12487

Variational quantum algorithms for graph state entanglement
Physical Review A 110, 032428 (2024)
APS Journals | arXiv:2406.10662

Numerical Methodologies

This project implements three solvers for the simulation of noisy quantum evolution.

1. Exact Density Matrix (DM)

Formalism: Solves the exact Liouville-von Neumann evolution for the mixed state $\rho$.

Implementation: Represents the state as a dense matrix of dimension $2^N \times 2^N$. Noise channels are applied via the Kraus Operator sum representation: $\rho' = \sum_k K_k \rho K_k^\dagger$.

Numerical Technique: Utilizes a tensor reshaping strategy $(2, 2, \dots, 2)_{bra} \otimes (2, 2, \dots, 2)_{ket}$ to avoid constructing full sparse matrices for local operations.

Computational Complexity: Memory scales as $O(4^N)$. This method provides the exact reference solution but is strictly limited to small systems ($N \lesssim 12$).

2. Tensor Network (TN)

Formalism: Represents the quantum channel $\Lambda$ and state evolution as a contraction of tensors.

Implementation: The noise channel is vectorized into a rank-4 Superoperator Tensor $T_{abcd} = \sum_k (K_k)_{ab} (K_k^*)_{cd}$. The expectation value $\mathcal{E}$ is computed by contracting the network of superoperators along the graph structure.

Computational Complexity: Memory scaling is linear $O(N)$ for low-treewidth graphs (e.g., 1D chains), but contraction time scales exponentially with the entanglement entropy (treewidth) of the graph.

3. Monte Carlo Wavefunction (MC) / Quantum Trajectories

Formalism: Stochastic unravelling of the Lindblad Master Equation.

Implementation: Evolves a single pure state $|\psi\rangle$ of dimension $2^N$. Dissipative processes defined by $\Lambda$ are modeled as stochastic Quantum Jumps determined by Monte Carlo sampling.

Numerical Technique: Leveraging JAX's vmap, thousands of independent trajectories are executed in parallel on GPU/TPU to estimate the ensemble average: $\rho \approx \frac{1}{M} \sum_{i=1}^M |\psi_i\rangle\langle\psi_i|$.

Computational Complexity: Memory scales as $O(2^N)$. This allows for the simulation of significantly larger systems ($N \approx 20-30$) compared to the DM approach, subject to statistical sampling error $\propto 1/\sqrt{M}$.

Installation

The framework relies on JAX for Just-In-Time (JIT) compilation and Automatic Differentiation (AD) to optimize the variational parameters $\vec{\theta}$.

# Clone the repository
git clone [https://github.com/yourusername/bell_correlations_in_mixed_graph_states.git](https://github.com/yourusername/bell_correlations_in_mixed_graph_states.git)
cd bell_correlations_in_mixed_graph_states

# Install requirements
pip install numpy pandas networkx matplotlib jax jaxlib optax


(Note: For GPU acceleration, please install the appropriate CUDA-enabled version of JAX).

Simulation & Usage

The workflow is divided into simulation (generating data) and analysis (visualization).

1. Simulation (bell_correlation_simulation.py)

This script executes the variational optimization loop. It sweeps over the noise parameter $p$ in channel $\Lambda$, optimizes the measurement angles $\vec{\theta}$ using the Adam optimizer, and computes the expectation value $\mathcal{E}$ and Bell correlation $Q$.

python bell_correlation_simulation.py


Configuration:
Modify the if __name__ == "__main__": block to define the physical system:

N: System size (number of qubits).

graph_type: Topology of the entangled state (star, grid, turan, etc.).

noise_type_vec: List of quantum channels to investigate (e.g., amplitude_damping, depolarizing).

Output:
Data is serialized into Pandas DataFrames and saved as .pkl files in the results/ directory, containing optimal angles, metrics ($\mathcal{E}, Q$), and topology data.

2. Visualization (plot_results.py)

This script generates publication-quality figures comparing the results of the different solvers and noise models.

python plot_results.py


It creates two types of plots:

Topology Visualization: Renders the graph structure (e.g., 2D Grid lattice).

Physics Comparison: Plots the Bell parameter $Q$ vs. noise probability $p$.

Verification Strategy

To ensure numerical validity, the visualization script employs a layered plotting style:

Exact DM: Plotted as large Hollow Circles.

Tensor Network: Plotted as small Solid Dots.

Monte Carlo: Plotted as Crosses with Error Bars representing the standard error of the mean (SEM).

Agreement: If the methods are consistent, the Solid Dots (TN) will lie perfectly within the Hollow Circles (DM), and the Crosses (MC) will align within statistical uncertainty.

Supported Graph Topologies

The framework supports the generation of various multipartite entanglement structures relevant to condensed matter and quantum information:

Star Graph: Highly connected central node (Greenberger–Horne–Zeilinger type entanglement).

Grid Graph: 2D Lattice structures ($m \times n$), relevant for solid-state implementations.

Turán Graph: Multipartite graphs $T(N, r)$.

1D Chain & Cycle: Linear cluster states.

Complete Graph: All-to-all connectivity.


License

This project is open-source under the MIT License.
