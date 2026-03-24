# Setting Up the Environment

```{objectives}
- Connect to your VM via SSH
- Initialize a fresh VM with required packages
- Clone the repository and install the phlearn package
- Start Jupyter Lab with SSH tunneling
- Verify PyTorch and GPU access
```

## 1. Connect to Your VM

Connect to your VM using SSH (see Episode 02 for Windows-specific instructions):

````{tabs}
```{tab} macOS / Linux / Git Bash
chmod 600 /path/to/your-key.pem
ssh -i /path/to/your-key.pem ubuntu@<VM_IP>
```

```{tab} Windows (PowerShell)
ssh -i "C:\Users\YourName\Downloads\your-key.pem" ubuntu@<VM_IP>
```
````

```{note}
**Windows users**: If you see "Permissions for key are too open", fix the key permissions first. See Episode 02, Step 7 for detailed instructions. Git Bash is recommended -- it supports `chmod` natively.
```

## 2. System Setup (Fresh VM)

On a fresh NAIC VM, install required system packages:

```bash
sudo apt update -y
sudo apt install -y build-essential git python3-dev python3-venv python3-pip libssl-dev zlib1g-dev
```

Alternatively, the repository includes a `vm-init.sh` script that automates system setup:

```bash
curl -O https://raw.githubusercontent.com/NAICNO/wp7-UC3-pseudo-hamiltonian-neural-networks/main/vm-init.sh
chmod +x vm-init.sh
./vm-init.sh
```

This will detect if module system (EasyBuild/Lmod) is available, install system packages if needed, and check GPU availability.

## 3. Clone and Setup

```bash
git clone https://github.com/NAICNO/wp7-UC3-pseudo-hamiltonian-neural-networks.git
cd pseudo-hamiltonian-neural-networks
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

The `setup.sh` script automatically:
1. Loads the Python module (if available via Lmod)
2. Validates Python version (3.8+ required)
3. Checks GPU and sets up CUDA symlinks
4. Creates a Python virtual environment
5. Installs the `phlearn` package in editable mode (`pip install -e phlearn`)
6. Installs dependencies from `requirements.txt`
7. Verifies PyTorch installation

## 4. Quick Verification

```bash
# Verify phlearn is importable
python -c "import phlearn; print('phlearn OK')"
python -c "import phlearn.phnns as phnn; print('phnns OK')"
python -c "import phlearn.phsystems.ode as phsys; print('phsystems OK')"

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 5. Start Jupyter Lab (Optional)

For interactive exploration, start Jupyter Lab inside a tmux session for persistence:

```bash
# Use tmux for persistence
tmux new -s jupyter
cd ~/pseudo-hamiltonian-neural-networks
source venv/bin/activate
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 
# Detach with Ctrl+B, then D
```

## 6. Create SSH Tunnel (on your local machine)

To access Jupyter Lab from your local browser, create an SSH tunnel. Open a **new terminal** on your local machine (not the VM):

````{tabs}
```{tab} macOS / Linux / Git Bash
# Verbose mode (recommended - shows connection status)
ssh -v -N -L 8888:localhost:8888 -i /path/to/your-key.pem ubuntu@<VM_IP>
```

```{tab} Windows (PowerShell)
ssh -v -N -L 8888:localhost:8888 -i "C:\Users\YourName\Downloads\your-key.pem" ubuntu@<VM_IP>
```
````

> **Note:** The tunnel will appear to "hang" after connecting -- this is normal! It means the tunnel is active. Keep the terminal open while using Jupyter.

**If port 8888 is already in use**, use an alternative port:

```bash
ssh -v -N -L 9999:localhost:8888 -i /path/to/your-key.pem ubuntu@<VM_IP>
# Then access via http://localhost:9999
```

Then navigate to: **http://localhost:8888/lab/tree/demonstrator-v1.orchestrator.ipynb**

To close the tunnel, press `Ctrl+C` in the terminal.

## Project Structure

After cloning, you will have:

```
pseudo-hamiltonian-neural-networks/
├── phlearn/                     # SINTEF phlearn package
│   ├── setup.py
│   └── phlearn/
│       ├── phnns/               # Neural network architectures
│       ├── phsystems/           # System simulators
│       │   ├── ode/             # Mass-spring, tank systems
│       │   └── pde/             # KdV, Cahn-Hilliard, BBM, etc.
│       ├── control/             # Model predictive control
│       └── utils/               # Utilities
├── example_scripts/             # SINTEF example notebooks
├── content/                     # Sphinx docs (this site)
├── setup.sh
├── vm-init.sh
├── demonstrator-v1.orchestrator.ipynb
├── requirements.txt
└── requirements-docs.txt
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: phlearn` | Run `pip install -e phlearn` from repo root |
| CUDA not available | Install GPU drivers: `nvidia-smi` to verify |
| SSH connection refused | Check key permissions: `chmod 600 /path/to/key.pem` |
| SSH "Permissions too open" (Windows) | Use Git Bash (`chmod 600`) or fix via icacls -- see Episode 02 |
| SSH connection timed out | Your IP may not be whitelisted -- add it at orchestrator.naic.no |
| Port 8888 already in use | Use alternative port: `-L 9999:localhost:8888` |
| `pip install` fails | Update pip: `pip install --upgrade pip` |
| Jupyter notebook won't start | Check installation: `pip install jupyterlab` |
| SSH tunnel appears to hang | This is normal -- tunnel is active, keep terminal open |
| Host key verification failed | Remove old key: `ssh-keygen -R <VM_IP>` |

```{keypoints}
- Set SSH key permissions with `chmod 600` before connecting (use Git Bash on Windows)
- Initialize fresh VMs with `vm-init.sh` or manual apt install
- Clone this repository directly -- all code and data are included
- Run `./setup.sh` to automatically set up the Python environment and install phlearn
- Verify with `python -c "import phlearn; print('OK')"` after setup
- Use tmux for persistent Jupyter Lab sessions
- Create an SSH tunnel to access Jupyter from your local browser
- Windows users: Git Bash is recommended for the best experience with SSH and Unix commands
```
