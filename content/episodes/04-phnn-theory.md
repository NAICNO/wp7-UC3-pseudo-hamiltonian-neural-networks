# PHNN Theory and Port-Hamiltonian Formulation

```{objectives}
- Understand the port-Hamiltonian formulation for dynamical systems
- Learn how each sub-network (S, R, F) enforces physical constraints
- Understand symmetric integration and why it matters
- Know how separation of concerns enables modular inference
```

```{admonition} Prerequisites
:class: note

This episode covers the mathematical foundations. You do not need to understand every equation to use the demonstrator -- but understanding the structure will help you interpret results.
```

## Port-Hamiltonian Systems

A pseudo-Hamiltonian system describes the time evolution of a state vector `x` as:

```
dx/dt = (S(x) - R(x)) * grad_H(x) + F(x)
```

Each term has a precise physical meaning:

| Term | Name | Property | Physical Role |
|------|------|----------|---------------|
| H(x) | Hamiltonian | Scalar function | Total energy of the system |
| S(x) | Structure matrix | Skew-symmetric | Energy-conserving coupling |
| R(x) | Dissipation matrix | Positive semi-definite | Energy dissipation (damping) |
| F(x) | External force | Unconstrained | External energy input/output |

The key insight: `S` being skew-symmetric guarantees that `x^T S x = 0`, so the conservative part never creates or destroys energy. `R` being positive semi-definite guarantees that energy only decreases through dissipation.

## Neural Network Architecture

Each term is parameterized by a separate neural network with built-in structural constraints:

```{mermaid}
graph TD
    X[State x] --> H_NET[H-network<br/>Energy estimator]
    X --> S_NET[S-network<br/>Skew-symmetric]
    X --> R_NET[R-network<br/>Pos. semi-definite]
    X --> F_NET[F-network<br/>External force]

    H_NET --> GRAD[grad H]
    S_NET --> STRUCT["(S - R)"]
    R_NET --> STRUCT

    STRUCT --> MULT["(S - R) * grad H"]
    MULT --> SUM["+"]
    F_NET --> SUM
    SUM --> DXDT["dx/dt"]

    style S_NET fill:#d4edda
    style R_NET fill:#fff3cd
    style F_NET fill:#d1ecf1
    style H_NET fill:#f8d7da
```

### S-network (Skew-Symmetric)

The S-network outputs a matrix that is anti-symmetrized:

```
S_out = A - A^T
```

This guarantees `S = -S^T` by construction, so energy is always conserved through this channel.

### R-network (Positive Semi-Definite)

The R-network outputs a matrix that passes through a positive semi-definite projection:

```
R_out = B * B^T
```

This guarantees `x^T R x >= 0`, so energy can only decrease through dissipation -- never increase.

### F-network (External Force)

The F-network has no structural constraint. It learns arbitrary state-dependent forcing that can inject or remove energy from the system.

## Separation of Concerns

```{mermaid}
graph LR
    TRAINED[Trained PHNN] --> INF1["Inference: Full system<br/>(S - R) grad H + F"]
    TRAINED --> INF2["Inference: Free system<br/>(S - R) grad H"]
    TRAINED --> INF3["Inference: New force<br/>(S - R) grad H + F'"]

    style INF1 fill:#d4edda
    style INF2 fill:#fff3cd
    style INF3 fill:#d1ecf1
```

Because each sub-network has a physical interpretation, you can:

- **Remove the external force** at inference time to study the free (unforced) system
- **Swap in a different forcing function** without retraining the model
- **Inspect the learned Hamiltonian** to verify the energy landscape is plausible
- **Analyze dissipation** to check which states lose energy and at what rate

This is impossible with a standard black-box neural network.

## Symmetric Integration

Training uses a **symmetric fourth-order integration scheme** rather than standard Euler or RK4. This matters because:

- Lower-order integrators (Euler, RK4) introduce systematic bias in energy estimates
- With sparse or noisy training data, this bias corrupts the learned Hamiltonian
- Symmetric integrators preserve the geometric structure of Hamiltonian systems

The phlearn package provides `midpoint` and `symmetric` integrators via the `train()` function.

## Energy Budget

For a well-trained PHNN, the energy budget at each time step satisfies:

```
dH/dt = -grad_H^T * R * grad_H + grad_H^T * F
         \___ dissipation ___/   \__ forcing __/
```

- Without external force: `dH/dt <= 0` (energy always decreases or stays constant)
- With external force: energy can increase or decrease depending on F

This is the fundamental guarantee that makes PHNNs physically valid.

```{keypoints}
- Port-Hamiltonian systems decompose dynamics into conservative, dissipative, and external force terms
- The S-network is anti-symmetrized by construction, guaranteeing energy conservation
- The R-network is projected to positive semi-definite, guaranteeing non-negative dissipation
- Symmetric integration preserves geometric structure during training
- Separation of concerns allows modifying forces at inference time without retraining
- The energy budget dH/dt = -grad_H^T R grad_H + grad_H^T F is guaranteed by architecture
```
