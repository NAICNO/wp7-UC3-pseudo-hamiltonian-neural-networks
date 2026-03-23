# Running the Demonstrator

```{objectives}
- Run the demonstrator notebook step by step
- Use example scripts for command-line training
- Set up tmux for background training runs
- Tune hyperparameters for different systems
```

## Demonstrator Notebook

The main entry point is `demonstrator-v1.orchestrator.ipynb`, designed to run on NAIC Orchestrator VMs.

### Starting the Notebook

```bash
cd ~/pseudo-hamiltonian-neural-networks
source venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token='' --NotebookApp.password=''
```

Then open **http://localhost:8888/lab/tree/demonstrator-v1.orchestrator.ipynb** via your SSH tunnel (see Episode 03).

### Notebook Walkthrough

The demonstrator covers:

1. **Environment verification** -- checks phlearn, PyTorch, and GPU availability
2. **System definition** -- sets up the mass-spring damper with configurable parameters
3. **Data generation** -- creates training trajectories using phlearn simulators
4. **PHNN training** -- trains the pseudo-Hamiltonian model with midpoint integrator
5. **Baseline training** -- trains a standard neural network for comparison
6. **Short-horizon evaluation** -- both models fit well on training-length predictions
7. **Long-horizon evaluation** -- PHNN maintains validity while baseline diverges
8. **Energy analysis** -- visualizes the learned Hamiltonian and dissipation

```{admonition} Execution Time
:class: tip

On a GPU VM (L40S), the full notebook runs in approximately 5-10 minutes. On CPU, expect 15-30 minutes. You can reduce training epochs for faster exploration.
```

## Example Scripts

For command-line training without Jupyter:

```bash
cd ~/pseudo-hamiltonian-neural-networks
source venv/bin/activate

# Train a PHNN model
python example_scripts/train_model.py

# Evaluate a trained model
python example_scripts/model_evaluation.py
```

### Available Example Notebooks

| Notebook | System | Time (GPU) |
|----------|--------|------------|
| `spring_example.ipynb` | Mass-spring damper | ~2 min |
| `phnn_ode_examples.ipynb` | Various ODE systems | ~5 min |
| `phnn_pde_examples.ipynb` | Various PDE systems | ~15 min |
| `kdv_example.ipynb` | KdV equation | ~10 min |
| `cahn_hilliard_example.ipynb` | Cahn-Hilliard equation | ~10 min |
| `bbm_example.ipynb` | BBM equation | ~10 min |
| `kdv_burgers_example.ipynb` | KdV-Burgers equation | ~10 min |

## Background Training with tmux

For longer training runs, use tmux to keep the process alive after disconnecting:

```bash
# Start a named tmux session
tmux new -s phnn

# Inside tmux:
cd ~/pseudo-hamiltonian-neural-networks
source venv/bin/activate
python example_scripts/train_model.py 2>&1 | tee training.log

# Detach: Ctrl+B, then D
```

To monitor and reattach:

```bash
# Monitor the log file
tail -f ~/pseudo-hamiltonian-neural-networks/training.log

# Reattach to the tmux session
tmux attach -t phnn
```

## Hyperparameter Tuning

Key hyperparameters to adjust:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `epochs` | 30 | More epochs improve fit but risk overfitting |
| `lr` (learning rate) | 0.001 | Lower values converge more slowly but more stably |
| `ntrajectories` | 300 | More training trajectories improve generalization |
| `integrator` | `midpoint` | `midpoint` for speed, `symmetric` for accuracy |
| `nstates` | 2 | System-dependent (2 for mass-spring, more for coupled systems) |

```{admonition} When to Use GPU
:class: note

For ODE systems (2-4 states, <1000 trajectories), CPU training is fast enough. GPU acceleration provides the most benefit for PDE systems with large spatial grids (64+ points) or when training many epochs.
```

```{keypoints}
- The demonstrator notebook is the primary entry point for interactive exploration
- Use tmux for background training to survive SSH disconnections
- Example scripts provide command-line alternatives to Jupyter
- ODE training is fast (minutes); PDE training benefits from GPU acceleration
- Key hyperparameters: epochs, learning rate, number of trajectories, integrator choice
- The `midpoint` integrator balances speed and accuracy for most use cases
```
