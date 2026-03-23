# Pseudo-Hamiltonian Neural Networks

Physics-preserving neural networks for dynamical systems that respect energy conservation, dissipation, and external forcing.

![PHNN vs Baseline: Long-Horizon Extrapolation](content/images/phnn_hero.png)

> **Key finding**: Decomposing system dynamics into Hamiltonian, dissipative, and external-force sub-networks produces models that remain physically valid even when external forces are modified or removed -- something standard neural networks cannot do.

## Table of Contents

1. [Overview](#overview)
2. [Methodology](#methodology)
3. [Supported Systems](#supported-systems)
4. [Sample Results](#sample-results)
5. [Getting Started](#getting-started)
6. [Example Notebooks](#example-notebooks)
7. [NAIC Orchestrator VM Deployment](#naic-orchestrator-vm-deployment)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)
10. [License](#license)

---

## Overview

Standard neural networks trained on physical systems drift into physically impossible states over long time horizons. They have no built-in notion of energy conservation, dissipation, or external forcing, so a model trained on a damped pendulum may predict perpetual motion or energy creation.

Pseudo-Hamiltonian Neural Networks (PHNNs) fix this by decomposing system dynamics into three physically meaningful components:

- **Conservative sub-network** -- captures energy-preserving Hamiltonian dynamics (symplectic structure)
- **Dissipative sub-network** -- models energy loss through damping, friction, or heat transfer
- **External force sub-network** -- learns state-dependent forcing terms that can be modified or removed at inference time

Each component is rooted in port-Hamiltonian theory and is physically interpretable. The framework supports both ODEs (mechanical systems, circuits) and PDEs (wave equations, reaction-diffusion).

The [`phlearn`](https://pypi.org/project/phlearn/) package, developed by SINTEF Digital, provides the reference implementation.

### What This Repository Includes

| Component | Description |
|-----------|-------------|
| **phlearn package** | SINTEF's pseudo-Hamiltonian neural network library (ODE + PDE support) |
| **ODE examples** | Mass-spring systems with damping and external forcing |
| **PDE examples** | KdV, Cahn-Hilliard, BBM, and KdV-Burgers equations |
| **MPC example** | Model predictive control using learned PHNN models |
| **Demonstrator notebook** | Interactive NAIC Orchestrator notebook ([`demonstrator-v1.orchestrator.ipynb`](demonstrator-v1.orchestrator.ipynb)) |

## Methodology

### 1. Port-Hamiltonian Formulation

A pseudo-Hamiltonian system takes the form:

```
dx/dt = (S(x) - R(x)) * grad_H(x) + F(x)
```

Where:
- `H(x)` is the Hamiltonian (total energy)
- `S(x)` is skew-symmetric (energy-conserving structure)
- `R(x)` is positive semi-definite (dissipation)
- `F(x)` is the external force

### 2. Neural Network Architecture

Each term is parameterized by a separate neural network with built-in structural constraints:

- The **S-network** outputs are anti-symmetrized, guaranteeing energy conservation
- The **R-network** outputs pass through a positive semi-definite projection, guaranteeing non-negative dissipation
- The **F-network** has no structural constraint, learning arbitrary state-dependent forcing

### 3. Symmetric Integration

Training uses a symmetric fourth-order integration scheme rather than standard Euler or RK4. This matters most with sparse or noisy data, where lower-order integrators introduce systematic bias.

### 4. Separation of Concerns

Because each sub-network has a physical interpretation, you can:
- Remove the external force at inference time to study the free system
- Swap in a different forcing function without retraining
- Inspect the learned Hamiltonian to verify energy landscape plausibility

## Supported Systems

### Ordinary Differential Equations

| System | Description |
|--------|-------------|
| Mass-spring | Forced and damped harmonic oscillator |
| Coupled oscillators | Multi-body Hamiltonian systems |
| Nonlinear pendulum | Large-angle pendulum with friction |

### Partial Differential Equations

| System | Description |
|--------|-------------|
| KdV (Korteweg-de Vries) | Shallow water waves, soliton dynamics |
| Cahn-Hilliard | Phase separation, spinodal decomposition |
| BBM (Benjamin-Bona-Mahony) | Regularized long waves |
| KdV-Burgers | Dispersive-diffusive wave propagation |

## Sample Results

PHNNs outperform standard neural networks on dynamical system benchmarks. The structural constraints prevent energy drift and keep trajectories physical over long integration windows.

On forced and damped mass-spring systems, PHNNs:
- Track ground-truth trajectories for 10x longer than unconstrained baselines
- Remain valid when external forces are modified post-training
- Learn Hamiltonians that match the analytical energy function

On PDE benchmarks (KdV, Cahn-Hilliard), the framework:
- Preserves conserved quantities (mass, energy) to machine precision
- Handles both periodic and non-periodic boundary conditions
- Scales to spatial grids of 256+ points

## Getting Started

### Project Structure

```
pseudo-hamiltonian-neural-networks/
├── README.md                                 # This file
├── AGENT.md / AGENT.yaml                     # AI agent setup instructions
├── LICENSE                                   # Dual license (CC BY-NC 4.0 + GPL-3.0)
├── .github/workflows/           # CI + Pages workflows (Sphinx Pages)
├── Makefile / make.bat                       # Sphinx build commands
├── setup.sh / vm-init.sh                     # Environment + VM setup
├── requirements.txt                          # Python dependencies
├── requirements-docs.txt                     # Sphinx documentation deps
├── widgets.py                                # Jupyter interactive widgets
├── utils.py                                  # Cluster/SSH utilities
├── demonstrator-v1.orchestrator.ipynb        # NAIC demonstrator notebook
├── phlearn/                                  # SINTEF phlearn package (MIT)
│   ├── setup.py                              # Package setup
│   └── phlearn/                              # Library source
│       ├── phnns/                            # Neural network architectures
│       ├── phsystems/                        # System simulators (ODE + PDE)
│       ├── control/                          # Model predictive control
│       └── utils/                            # Utilities
├── example_scripts/                          # SINTEF example notebooks
│   ├── phnn_ode_examples.ipynb               # ODE tutorial
│   ├── phnn_pde_examples.ipynb               # PDE tutorial
│   ├── spring_example.ipynb                  # Mass-spring demo
│   ├── kdv_example.ipynb                     # KdV equation
│   ├── cahn_hilliard_example.ipynb           # Cahn-Hilliard equation
│   ├── bbm_example.ipynb                     # BBM equation
│   ├── kdv_burgers_example.ipynb             # KdV-Burgers equation
│   └── pm_example.ipynb                      # Porous medium example
├── content/                                  # Sphinx documentation site
│   ├── conf.py                               # Sphinx config
│   ├── index.rst                             # Documentation root
│   ├── episodes/                             # Tutorial chapters (8 episodes)
│   └── images/                               # Result visualizations
├── scripts/                                  # Utility scripts
│   └── generate_images.py                    # Generate documentation images
├── tests/                                    # Test suite (69 tests)
│   └── test_demonstrator.py                  # PHNN demonstrator tests
└── results/                                  # Training output directory
```

### Installation

**NAIC Orchestrator VM** (recommended):

```bash
# 1. SSH into your VM
ssh -i ~/.ssh/naic-vm.pem ubuntu@<YOUR_VM_IP>

# 2. Init VM (first time only)
curl -O https://raw.githubusercontent.com/NAICNO/wp7-UC3-pseudo-hamiltonian-neural-networks/main/vm-init.sh
chmod +x vm-init.sh && ./vm-init.sh

# 3. Clone and setup
git clone git@github.com:NAICNO/wp7-UC3-pseudo-hamiltonian-neural-networks.git
cd pseudo-hamiltonian-neural-networks
chmod +x setup.sh && ./setup.sh
source venv/bin/activate

# 4. Run the demonstrator
jupyter lab demonstrator-v1.orchestrator.ipynb
```

**Local machine:**

```bash
git clone git@github.com:NAICNO/wp7-UC3-pseudo-hamiltonian-neural-networks.git
cd pseudo-hamiltonian-neural-networks
python3 -m venv venv && source venv/bin/activate
pip install -e phlearn
pip install -r requirements.txt
jupyter lab demonstrator-v1.orchestrator.ipynb
```

### Quick Start

```python
import numpy as np
import phlearn.phsystems.ode as phsys
import phlearn.phnns as phnn

# 1. Define a damped mass-spring system
nstates = 2
R = np.diag([0, 0.3])  # damping on momentum only
M = np.diag([0.5, 0.5])  # H(q,p) = k/2*q^2 + 1/(2m)*p^2

system = phsys.PseudoHamiltonianSystem(
    nstates=nstates,
    hamiltonian=lambda x: x.T @ M @ x,
    grad_hamiltonian=lambda x: 2 * M @ x,
    dissipation_matrix=R,
)

# 2. Generate training data
t_axis = np.linspace(0, 10, 101)
traindata = phnn.generate_dataset(system, ntrajectories=300, t_sample=t_axis)

# 3. Build and train a PHNN
states_dampened = np.diagonal(R) != 0
model = phnn.PseudoHamiltonianNN(
    nstates, dissipation_est=phnn.R_estimator(states_dampened)
)
model, _ = phnn.train(model, integrator='midpoint', traindata=traindata, epochs=30)

# 4. Predict -- energy structure is guaranteed by architecture
x_pred, _ = model.simulate_trajectory(integrator=False, t_sample=t_axis, x0=[1.0, 0.0])
```

## Example Notebooks

| Notebook | System | Description |
|----------|--------|-------------|
| `phnn_ode_examples.ipynb` | Various ODEs | Overview of PHNN applied to mechanical systems |
| `phnn_pde_examples.ipynb` | Various PDEs | PHNN for spatially extended systems |
| `spring_example.ipynb` | Mass-spring | Damped oscillator with external forcing |
| `kdv_example.ipynb` | KdV equation | Soliton propagation and interaction |
| `cahn_hilliard_example.ipynb` | Cahn-Hilliard | Phase separation dynamics |
| `bbm_example.ipynb` | BBM equation | Regularized long wave propagation |
| `kdv_burgers_example.ipynb` | KdV-Burgers | Combined dispersion and diffusion |
| `pm_example.ipynb` | Porous medium | Flow through porous media |

## NAIC Orchestrator VM Deployment

### Jupyter Access

```bash
# On VM:
jupyter lab --no-browser --ip=0.0.0.0 --port=8888

# On laptop (SSH tunnel):
ssh -N -L 8888:localhost:8888 -i ~/.ssh/naic-vm.pem ubuntu@<YOUR_VM_IP>
```

Open: **http://localhost:8888**

### Background Training

```bash
tmux new-session -d -s phnn 'cd ~/pseudo-hamiltonian-neural-networks && \
  source venv/bin/activate && \
  python example_scripts/train_model.py \
  2>&1 | tee training.log'

tail -f training.log          # monitor
tmux attach -t phnn           # reattach
```

### Resources

- NAIC Portal: https://orchestrator.naic.no
- VM Workflows Guide: https://training.pages.sigma2.no/tutorials/naic-cloud-vm-workflows/
- phlearn Documentation: https://pseudo-hamiltonian-neural-networks.readthedocs.io/en/latest/

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SSH permission denied | `chmod 600 ~/.ssh/naic-vm.pem` |
| SSH timeout | Verify VM IP in orchestrator.naic.no |
| ModuleNotFoundError: phlearn | `pip install phlearn` or `pip install -e phlearn` from source |
| CUDA out of memory | Reduce batch size or spatial grid resolution |
| Notebook kernel dies | Check available RAM with `free -h`; restart kernel |
| Integration diverges | Reduce time step or use smaller learning rate |
| Host key error | `ssh-keygen -R <VM_IP>` |

## References

- Eidnes, S., Stasik, A.J., Sterud, C., Benth, E., and Lye, K.O. (2023). Pseudo-Hamiltonian neural networks for learning partial differential equations. *Journal of Computational Physics*, 500, 112738.
- Eidnes, S. and Lye, K.O. (2024). Pseudo-Hamiltonian neural networks with state-dependent external forces. *Applied Mathematics and Computation*, 459, 128242.
- SINTEF Digital -- phlearn package: https://github.com/SINTEF/pseudo-hamiltonian-neural-networks

## Contributors

- **Sølve Eidnes** (SINTEF Digital) -- Lead developer, PHNN theory and implementation
- **Kjetil Olsen Lye** (SINTEF Digital) -- PDE extensions, co-author

## AI Agent

If using an AI coding assistant, see [`AGENT.md`](AGENT.md) for automated setup instructions.

## License

This project uses a dual license:

- **Tutorial content** (`*.md`, `*.ipynb`): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- **Software code** (`*.py`, `*.sh`): [GPL-3.0-only](https://www.gnu.org/licenses/gpl-3.0.txt)

The upstream `phlearn` package is licensed under MIT by SINTEF.

Copyright (c) 2026 Sigma2 / NAIC. See [LICENSE](LICENSE) for full details.
