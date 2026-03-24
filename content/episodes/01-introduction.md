# Introduction to Pseudo-Hamiltonian Neural Networks

```{objectives}
- Understand what Pseudo-Hamiltonian Neural Networks (PHNNs) are
- Learn why physics-preserving structure matters for dynamical systems
- Know the 3-component decomposition: conservative + dissipative + external force
- Understand the difference between PHNNs and standard neural networks
- Know the repository structure and key references
```

```{admonition} Why This Matters
:class: tip

**The Scenario:** An engineer simulating a damped mechanical system notices that a standard neural network predicts energy *increasing* over time -- a physical impossibility for a dissipative system. The model fits short-horizon training data well but diverges on longer predictions.

**The Research Question:** Can we build neural networks that respect energy conservation and dissipation by design, so that long-horizon predictions remain physically valid -- even when external forces change?

**What This Episode Gives You:** The big picture -- how PHNNs decompose dynamics into three physically meaningful sub-networks, why this architecture outperforms standard approaches, and what the repository contains.
```

## Overview

Standard neural networks trained on physical systems have no built-in notion of energy conservation or dissipation. Over long time horizons, they drift into physically impossible states -- predicting perpetual motion, energy creation, or unbounded trajectories.

**Pseudo-Hamiltonian Neural Networks (PHNNs)** solve this by decomposing system dynamics into three physically meaningful components:

- **Conservative sub-network** -- captures energy-preserving Hamiltonian dynamics (skew-symmetric structure)
- **Dissipative sub-network** -- models energy loss through damping, friction, or heat transfer (positive semi-definite)
- **External force sub-network** -- learns state-dependent forcing terms that can be modified or removed at inference time

Each component is rooted in port-Hamiltonian theory and is physically interpretable. The framework supports both ODEs (mechanical systems, circuits) and PDEs (wave equations, reaction-diffusion).

## PHNN vs Standard Neural Network

| Property | Standard NN | PHNN |
|----------|------------|------|
| Energy conservation | Not guaranteed | Built into architecture |
| Long-horizon stability | Degrades rapidly | Physically valid |
| Force modification | Requires retraining | Swap at inference time |
| Interpretability | Black box | Each sub-network has physical meaning |
| Training data needed | More | Less (physics provides inductive bias) |

## 3-Component Decomposition

The core equation of a pseudo-Hamiltonian system:

```
dx/dt = (S(x) - R(x)) * grad_H(x) + F(x)
```

Where:
- **H(x)** -- Hamiltonian (total energy), learned by the energy network
- **S(x)** -- Skew-symmetric matrix (energy-conserving structure), learned by the S-network
- **R(x)** -- Positive semi-definite matrix (dissipation), learned by the R-network
- **F(x)** -- External force, learned by the F-network

## Repository Structure

| Component | Location |
|-----------|----------|
| phlearn package | `phlearn/` (SINTEF's PHNN library) |
| ODE systems | `phlearn/phlearn/phsystems/ode/` |
| PDE systems | `phlearn/phlearn/phsystems/pde/` |
| Neural network architectures | `phlearn/phlearn/phnns/` |
| Example notebooks | `example_scripts/` |
| Demonstrator notebook | `demonstrator-v1.orchestrator.ipynb` |
| Setup scripts | `setup.sh`, `vm-init.sh` |

## Using AI Coding Assistants

If you are using an AI coding assistant, the repository includes an `AGENT.md` file with setup instructions. Tell your assistant:

> "Read AGENT.md and help me run the PHNN demonstrator on my NAIC VM."

## What You Will Learn

| Episode | Topic |
|---------|-------|
| 02 | Provisioning a NAIC VM |
| 03 | Setting up the environment |
| 04 | PHNN theory and port-Hamiltonian formulation |
| 05 | ODE systems: mass-spring, training, and comparison |
| 06 | Running the demonstrator notebook |
| 07 | PDE extensions: KdV, Cahn-Hilliard, BBM |
| 08 | FAQ and troubleshooting |

## References

- Eidnes, S., Stasik, A.J., Sterud, C., Benth, E., and Lye, K.O. (2023). Pseudo-Hamiltonian neural networks for learning partial differential equations. *Journal of Computational Physics*, 500, 112738.
- Eidnes, S. and Lye, K.O. (2024). Pseudo-Hamiltonian neural networks with state-dependent external forces. *Applied Mathematics and Computation*, 459, 128242.
- SINTEF Digital -- phlearn package: https://github.com/SINTEF/pseudo-hamiltonian-neural-networks

```{keypoints}
- PHNNs decompose dynamics into conservative, dissipative, and external force components
- Each sub-network has built-in structural constraints guaranteeing physical validity
- Standard NNs drift into impossible states; PHNNs remain valid over long horizons
- External forces can be modified or removed at inference time without retraining
- The phlearn package by SINTEF provides the reference implementation for both ODEs and PDEs
- All code, examples, and the demonstrator notebook are included in this repository
```
