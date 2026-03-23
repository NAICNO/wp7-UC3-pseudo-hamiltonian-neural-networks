# FAQ and Troubleshooting

```{objectives}
- Resolve common setup and runtime issues
- Understand PHNN-specific design choices
- Know how to extend the framework with new systems
```

## General Setup

**Q: What Python version is required?**

Python 3.8 or higher. The `setup.sh` script validates this automatically. We recommend Python 3.10+ for best compatibility with current PyTorch versions.

**Q: Do I need a GPU?**

No. All examples run on CPU. However, GPU acceleration (CUDA) significantly speeds up PDE training. For ODE examples, CPU is sufficient. The NAIC Orchestrator VMs provide L40S GPUs.

**Q: How do I check GPU availability?**

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## Common Errors

**Q: `ModuleNotFoundError: No module named 'phlearn'`**

The phlearn package must be installed from the local source:

```bash
cd ~/pseudo-hamiltonian-neural-networks
source venv/bin/activate
pip install -e phlearn
```

**Q: `CUDA out of memory`**

Reduce batch size or spatial grid resolution. For PDE systems, try a coarser grid (e.g., 64 instead of 256 points). You can also force CPU training:

```python
model = model.to('cpu')
```

**Q: Notebook kernel dies during training**

Check available RAM with `free -h`. If memory is low, reduce the number of training trajectories or close other notebooks. Restart the kernel and try again.

**Q: Integration diverges (NaN values)**

- Reduce the learning rate (e.g., from 0.001 to 0.0001)
- Reduce the time step in trajectory generation
- Use the `midpoint` integrator (more stable than `symmetric` for stiff systems)
- Check that initial conditions are within a reasonable range

**Q: `Host key verification failed` when SSHing**

A new VM was created at the same IP address. Remove the old key:

```bash
ssh-keygen -R <VM_IP>
```

## PHNN-Specific Questions

**Q: What is the difference between PHNN and HNN (Hamiltonian Neural Networks)?**

| Feature | HNN | PHNN |
|---------|-----|------|
| Dissipation | Not modeled | Explicit R-network |
| External forces | Not modeled | Explicit F-network |
| Scope | Conservative systems only | Conservative + dissipative + forced |
| Real-world applicability | Limited (no friction) | Broad (friction, damping, forcing) |

HNNs (Greydanus et al., 2019) only model conservative systems. PHNNs extend this to dissipative and forced systems, which covers most real-world applications.

**Q: How do I choose an integrator?**

| Integrator | Speed | Accuracy | Use Case |
|------------|-------|----------|----------|
| `midpoint` | Fast | Good | Default choice, most systems |
| `symmetric` | Slower | Best | Sparse/noisy data, long horizons |

Start with `midpoint`. Switch to `symmetric` if you observe energy drift or are working with sparse data.

**Q: Can I add a new ODE system?**

Yes. Create a new system class that defines:

1. The Hamiltonian function `H(x)`
2. The gradient `grad_H(x)`
3. The dissipation matrix `R(x)` (can be zero)
4. Optional external force `F(x)`

See `phlearn/phlearn/phsystems/ode/msd_system.py` for a reference implementation.

**Q: Can I add a new PDE system?**

Yes. Subclass `PseudoHamiltonianPDESystem` and define:

1. The Hamiltonian functional `H[u]`
2. The structure operator `S`
3. The dissipation operator `R` (if applicable)
4. Boundary conditions

See `phlearn/phlearn/phsystems/pde/kdv_system.py` for a reference implementation.

**Q: How do I save and load trained models?**

```python
import torch

# Save
torch.save(model.state_dict(), 'phnn_model.pt')

# Load
model.load_state_dict(torch.load('phnn_model.pt'))
model.eval()
```

## Resources

- phlearn documentation: https://pseudo-hamiltonian-neural-networks.readthedocs.io/en/latest/
- NAIC Portal: https://www.naic.no/
- Orchestrator: https://orchestrator.naic.no/
- Repository: https://github.com/NAICNO/wp7-UC3-pseudo-hamiltonian-neural-networks

```{keypoints}
- Python 3.8+ is required; GPU is optional but recommended for PDE training
- Install phlearn with `pip install -e phlearn` from the repository root
- PHNNs extend HNNs by adding dissipation (R-network) and external forces (F-network)
- Use `midpoint` integrator as default; switch to `symmetric` for sparse/noisy data
- New ODE/PDE systems can be added by subclassing the provided base classes
- Check `nvidia-smi` and `torch.cuda.is_available()` to verify GPU setup
```
