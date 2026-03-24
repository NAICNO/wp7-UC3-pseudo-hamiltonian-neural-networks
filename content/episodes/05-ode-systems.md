# ODE Systems: Mass-Spring and Beyond

```{objectives}
- Understand the mass-spring damper as a port-Hamiltonian system
- Generate training data using phlearn system simulators
- Train a PHNN and compare with a baseline neural network
- Evaluate short-horizon vs long-horizon prediction accuracy
- Observe damping recovery and external force learning
```

## Mass-Spring Damper System

The damped mass-spring is the canonical example for PHNNs. The state vector is `x = [q, p]` (position and momentum):

```
H(q, p) = k/2 * q^2 + 1/(2m) * p^2     (total energy)
S = [[0, 1], [-1, 0]]                     (symplectic structure)
R = [[0, 0], [0, d]]                      (damping on momentum)
```

With damping coefficient `d > 0`, the system dissipates energy through the momentum channel while the symplectic structure couples position and momentum.

## Generating Training Data

The phlearn package provides system simulators that generate ground-truth trajectories:

```python
import numpy as np
import phlearn.phsystems.ode as phsys
import phlearn.phnns as phnn

# Define system parameters
nstates = 2
R = np.diag([0, 0.3])           # damping on momentum only
M = np.diag([0.5, 0.5])         # mass and spring constant

system = phsys.PseudoHamiltonianSystem(
    nstates=nstates,
    hamiltonian=lambda x: x.T @ M @ x,
    grad_hamiltonian=lambda x: 2 * M @ x,
    dissipation_matrix=R,
)

# Generate 300 trajectories with random initial conditions
t_axis = np.linspace(0, 10, 101)
traindata = phnn.generate_dataset(system, ntrajectories=300, t_sample=t_axis)
```

## Training a PHNN

Building and training a PHNN requires specifying which states are damped:

```python
# Tell the model which states have dissipation
states_dampened = np.diagonal(R) != 0  # [False, True]

model = phnn.PseudoHamiltonianNN(
    nstates,
    dissipation_est=phnn.R_estimator(states_dampened),
)

# Train with symmetric midpoint integrator
model, losses = phnn.train(
    model,
    integrator='midpoint',
    traindata=traindata,
    epochs=30,
)
```

## PHNN vs Baseline Comparison

The demonstrator notebook trains both a PHNN and a standard neural network (baseline) on the same data. Key differences emerge on long-horizon predictions:

| Metric | Baseline NN | PHNN |
|--------|------------|------|
| Short-horizon error (1 period) | Low | Low |
| Long-horizon error (10 periods) | Grows rapidly | Stays bounded |
| Energy conservation | Violated | Guaranteed by architecture |
| Damping behavior | May predict energy gain | Always dissipates correctly |

### Short-Horizon vs Long-Horizon

- **Short horizon**: Both models fit well because the training data covers this range
- **Long horizon**: The baseline drifts because it has no energy structure; the PHNN remains physical because the architecture enforces energy constraints at every time step

## Damping Recovery

A key advantage of PHNNs: the R-network learns the dissipation structure from data. After training, you can inspect the learned `R` matrix to verify:

- Which states are damped (should match the true system)
- The magnitude of damping (should approximate the true damping coefficient)
- Off-diagonal terms (should be near zero for simple systems)

## External Force Learning

When training on a system with external forces, the F-network learns the forcing function separately:

```python
# System with external force
def external_force(x):
    return np.array([0, np.sin(x[0])])

system_forced = phsys.PseudoHamiltonianSystem(
    nstates=nstates,
    hamiltonian=lambda x: x.T @ M @ x,
    grad_hamiltonian=lambda x: 2 * M @ x,
    dissipation_matrix=R,
    external_forces=external_force,
)
```

After training, you can:
- Remove the force (`F = 0`) to simulate the free system
- Replace the force with a different function
- Analyze what the F-network learned

## Running the Examples

The `example_scripts/spring_example.ipynb` notebook walks through the full mass-spring example. The `example_scripts/phnn_ode_examples.ipynb` notebook covers additional ODE systems.

```{keypoints}
- The mass-spring damper is a canonical port-Hamiltonian system with symplectic structure and dissipation
- phlearn provides system simulators for generating ground-truth training data
- PHNNs maintain physical validity on long-horizon predictions where standard NNs diverge
- The R-network recovers the true dissipation structure from data
- External forces are learned separately and can be modified at inference time
- See `spring_example.ipynb` and `phnn_ode_examples.ipynb` for hands-on examples
```
