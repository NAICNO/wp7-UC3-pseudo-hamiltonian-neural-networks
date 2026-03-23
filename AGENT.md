# AI Agent Setup Instructions

## Quick Start

### Step 1: SSH into your NAIC VM

Replace the IP address with the one shown in the NAIC Orchestrator portal.
Do NOT type the angle brackets -- use the actual IP and key path.

```bash
# Example with a .pem key (common on NAIC):
ssh -i ~/.ssh/naic-vm.pem ubuntu@10.212.136.52

# Example with a standard key:
ssh -i ~/.ssh/id_rsa ubuntu@10.212.136.52

# If you get "Permission denied", check:
#   1. The key file has correct permissions:  chmod 600 ~/.ssh/naic-vm.pem
#   2. You are using the right username (ubuntu, not root)
#   3. The IP matches your VM in orchestrator.naic.no
```

### Step 2: Initialize VM (first time only)

```bash
curl -O https://raw.githubusercontent.com/NAICNO/wp7-UC3-pseudo-hamiltonian-neural-networks/main/vm-init.sh
chmod +x vm-init.sh
./vm-init.sh
```

### Step 3: Clone and setup

```bash
git clone https://github.com/NAICNO/wp7-UC3-pseudo-hamiltonian-neural-networks.git
cd pseudo-hamiltonian-neural-networks
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Step 4: Run the demonstrator

```bash
jupyter lab demonstrator-v1.orchestrator.ipynb
```

### Step 5: Run example scripts

```bash
cd example_scripts
jupyter lab spring_example.ipynb
```

## Jupyter Notebook Access

Start Jupyter on the VM, then create an SSH tunnel from your laptop.

**On the VM:**
```bash
cd pseudo-hamiltonian-neural-networks
source venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

**On your laptop** (new terminal -- replace IP and key path with yours):
```bash
ssh -v -N -L 8888:localhost:8888 -i ~/.ssh/naic-vm.pem ubuntu@10.212.136.52

# Then open in your browser:
#   http://localhost:8888
```

Common mistakes:
- Do NOT keep the angle brackets. `ubuntu@<VM_IP>` means type `ubuntu@10.212.136.52` (your actual IP).
- `-N` means "no remote command" -- the terminal will appear to hang. That is normal.
- `-v` enables verbose output so you can see connection progress.
- If port 8888 is already in use locally, pick another: `-L 9999:localhost:8888` then open `http://localhost:9999`.

## Verification Steps

1. Check Python: `python3 --version` (need 3.8+)
2. Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check phlearn: `python -c "import phlearn; print('OK')"`
4. Run demonstrator: `jupyter lab demonstrator-v1.orchestrator.ipynb`

## Project Structure

```
pseudo-hamiltonian-neural-networks/
├── AGENT.md                              # This file
├── README.md                             # Project overview
├── setup.sh                              # Environment setup
├── vm-init.sh                            # VM initialization
├── requirements.txt                      # Dependencies
├── LICENSE                               # Dual license (CC BY-NC 4.0 + GPL-3.0)
├── demonstrator-v1.orchestrator.ipynb    # NAIC demonstrator notebook
├── phlearn/                              # SINTEF phlearn package (MIT)
│   ├── setup.py                          # Package setup
│   ├── requirements.txt                  # Package dependencies
│   └── phlearn/                          # Library source
│       ├── phnns/                        # Neural network models
│       ├── phsystems/                    # System simulators (ODE + PDE)
│       ├── control/                      # Model predictive control
│       └── utils/                        # Utilities
└── example_scripts/                      # Notebooks and scripts
    ├── phnn_ode_examples.ipynb           # ODE tutorial
    ├── phnn_pde_examples.ipynb           # PDE tutorial
    ├── spring_example.ipynb              # Mass-spring demo
    ├── kdv_example.ipynb                 # KdV equation
    ├── cahn_hilliard_example.ipynb       # Cahn-Hilliard
    ├── bbm_example.ipynb                 # BBM equation
    ├── kdv_burgers_example.ipynb         # KdV-Burgers
    ├── pm_example.ipynb                  # Porous medium
    ├── train_model.py                    # Training script
    ├── model_evaluation.py               # Evaluation
    └── mpc_example.py                    # Model predictive control
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SSH "Permission denied (publickey)" | Check key permissions: `chmod 600 ~/.ssh/your-key.pem` |
| SSH hangs or times out | Verify VM IP in orchestrator.naic.no; check VPN if required |
| Typed `<VM_IP>` literally | Replace with your actual IP, e.g. `ubuntu@10.212.136.52` |
| Jupyter tunnel not working | Make sure `-N` flag is present; check port isn't already used |
| CUDA out of memory | Reduce batch size or spatial grid resolution |
| ModuleNotFoundError: phlearn | Run `pip install -e phlearn` from the project root |
| Permission denied (scripts) | `chmod +x setup.sh vm-init.sh` |
| No GPU detected | Check `nvidia-smi`; install CUDA drivers |
