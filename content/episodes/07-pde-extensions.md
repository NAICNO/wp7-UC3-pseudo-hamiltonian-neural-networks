# PDE Extensions

```{objectives}
- Understand how PHNNs extend from ODEs to PDEs
- Know the supported PDE systems: KdV, Cahn-Hilliard, BBM, KdV-Burgers
- Learn about conservation properties for spatially extended systems
- Run PDE example notebooks
```

## From ODEs to PDEs

The port-Hamiltonian framework extends naturally to spatially extended systems. Instead of a finite state vector `x`, PDE systems operate on fields `u(x, t)` discretized on a spatial grid.

The structure is the same:

```
du/dt = (S - R) * delta_H/delta_u + F
```

Where `S`, `R`, and `H` now operate on the discretized spatial field. The phlearn package handles spatial derivatives, boundary conditions, and grid discretization internally.

## Supported PDE Systems

### KdV (Korteweg-de Vries)

```
u_t + u * u_x + u_xxx = 0
```

- Models shallow water waves and soliton dynamics
- Conserves mass and energy in the Hamiltonian formulation
- See `example_scripts/kdv_example.ipynb`

### Cahn-Hilliard

```
u_t = div(M * grad(mu)),    mu = -eps^2 * laplacian(u) + f'(u)
```

- Models phase separation and spinodal decomposition
- Conserves total mass; free energy decreases monotonically
- See `example_scripts/cahn_hilliard_example.ipynb`

### BBM (Benjamin-Bona-Mahony)

```
u_t + u_x + u * u_x - u_xxt = 0
```

- Regularized alternative to KdV for long wave propagation
- Better dispersion properties than KdV for numerical computation
- See `example_scripts/bbm_example.ipynb`

### KdV-Burgers

```
u_t + u * u_x + u_xxx - nu * u_xx = 0
```

- Combines KdV dispersion with Burgers diffusion
- Models dispersive-diffusive wave propagation
- See `example_scripts/kdv_burgers_example.ipynb`

### Additional Systems

The phlearn package also includes:

| System | Module | Description |
|--------|--------|-------------|
| Heat equation | `heat_system.py` | Diffusion on 1D domain |
| Allen-Cahn | `allen_cahn_system.py` | Reaction-diffusion with bistable potential |
| Perona-Malik | `perona_malik_system.py` | Nonlinear diffusion (image processing) |

## Conservation Properties

A key advantage of the PHNN-PDE framework is that conservation laws are respected by the architecture:

| Property | Standard NN | PHNN-PDE |
|----------|------------|----------|
| Mass conservation | Approximate | Exact (skew-symmetric S) |
| Energy dissipation | Not guaranteed | Guaranteed (positive semi-definite R) |
| Soliton preservation | Degrades over time | Maintained |
| Long-time stability | Blows up | Bounded |

For the KdV equation, the PHNN preserves both mass and energy to near machine precision. For Cahn-Hilliard, total mass is conserved while free energy decreases monotonically.

## Running PDE Examples

```bash
cd ~/pseudo-hamiltonian-neural-networks
source venv/bin/activate
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

The comprehensive PDE tutorial is `example_scripts/phnn_pde_examples.ipynb`. Individual system notebooks provide focused examples for each equation.

```{admonition} GPU Recommended
:class: warning

PDE training involves larger state vectors (spatial grids of 64-256 points) and benefits significantly from GPU acceleration. On CPU, PDE examples may take 30-60 minutes; on GPU (L40S), they typically complete in 10-15 minutes.
```

## PDE System Implementation

The phlearn PDE systems are located in `phlearn/phlearn/phsystems/pde/`. Each system defines:

- The Hamiltonian functional H[u]
- The structure operator S (typically a spatial derivative operator)
- The dissipation operator R (if applicable)
- Boundary conditions (periodic or non-periodic)
- Spatial discretization parameters

```{keypoints}
- PHNNs extend naturally from ODEs to PDEs via discretized port-Hamiltonian operators
- Supported systems: KdV, Cahn-Hilliard, BBM, KdV-Burgers, heat, Allen-Cahn, Perona-Malik
- Conservation properties (mass, energy) are enforced by the network architecture
- PDE training benefits from GPU acceleration due to larger state vectors
- The `phnn_pde_examples.ipynb` notebook provides a comprehensive PDE tutorial
- Individual notebooks exist for each PDE system in `example_scripts/`
```
