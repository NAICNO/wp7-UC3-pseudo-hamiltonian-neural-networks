"""
Generate result images for README and Sphinx documentation.

Run from the repo root:
    python scripts/generate_images.py

Saves plots to content/images/.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import phlearn.phsystems.ode as phsys
import phlearn.phnns as phnn

ttype = torch.float32
torch.set_default_dtype(ttype)

OUTPUT_DIR = 'content/images'


def make_system(damping=0.3, external_forces=None):
    nstates = 2
    R = np.diag([0, damping])
    M = np.diag([0.5, 0.5])

    def hamiltonian(x):
        return x.T @ M @ x

    def hamiltonian_grad(x):
        return 2 * M @ x

    return phsys.PseudoHamiltonianSystem(
        nstates=nstates,
        hamiltonian=hamiltonian,
        grad_hamiltonian=hamiltonian_grad,
        dissipation_matrix=R,
        external_forces=external_forces,
    ), M


def train_models(system, epochs=30, ntrajectories=300, dt=0.1, tmax=10):
    nstates = 2
    t_axis = np.linspace(0, tmax, round(tmax / dt) + 1)
    traindata = phnn.generate_dataset(system, ntrajectories, t_axis)

    states_dampened = np.diagonal(system.dissipation_matrix) != 0
    phmodel = phnn.PseudoHamiltonianNN(
        nstates, dissipation_est=phnn.R_estimator(states_dampened)
    )
    baseline_nn = phnn.BaselineNN(nstates, hidden_dim=100)
    basemodel = phnn.DynamicSystemNN(nstates, baseline_nn)

    phmodel, _ = phnn.train(phmodel, integrator='midpoint', traindata=traindata, epochs=epochs, batch_size=32)
    basemodel, _ = phnn.train(basemodel, integrator='midpoint', traindata=traindata, epochs=epochs, batch_size=32)

    return phmodel, basemodel, t_axis


def generate_hero_image(phmodel, basemodel, system, M):
    """Generate the main comparison image for README."""
    t_long = np.linspace(0, 50, 501)
    x0 = [1.5, 0.5]

    x_exact, *_ = system.sample_trajectory(t_long, x0=x0)
    x_phnn, _ = phmodel.simulate_trajectory(integrator=False, t_sample=t_long, x0=x0)
    x_baseline, _ = basemodel.simulate_trajectory(integrator=False, t_sample=t_long, x0=x0)

    energy_exact = [x_exact[i].T @ M @ x_exact[i] for i in range(len(t_long))]
    energy_phnn = [x_phnn[i].T @ M @ x_phnn[i] for i in range(len(t_long))]
    energy_baseline = [x_baseline[i].T @ M @ x_baseline[i] for i in range(len(t_long))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(x_exact[:, 0], x_exact[:, 1], 'k--', linewidth=2, label='Exact', alpha=0.7)
    ax.plot(x_phnn[:, 0], x_phnn[:, 1], 'b-', linewidth=1.5, label='PHNN')
    ax.plot(x_baseline[:, 0], x_baseline[:, 1], 'r-', linewidth=1.5, label='Baseline NN', alpha=0.7)
    ax.plot(x0[0], x0[1], 'ko', markersize=8)
    ax.set_xlabel('Position (q)', fontsize=12)
    ax.set_ylabel('Momentum (p)', fontsize=12)
    ax.set_title('Phase Portrait (t = 0 to 50)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_long, x_exact[:, 0], 'k--', linewidth=2, label='Exact', alpha=0.7)
    ax.plot(t_long, x_phnn[:, 0], 'b-', linewidth=1.5, label='PHNN')
    ax.plot(t_long, x_baseline[:, 0], 'r-', linewidth=1.5, label='Baseline NN', alpha=0.7)
    ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5, label='Training horizon')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Position (q)', fontsize=12)
    ax.set_title('Position: Training vs Extrapolation', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_long, energy_exact, 'k--', linewidth=2, label='Exact', alpha=0.7)
    ax.plot(t_long, energy_phnn, 'b-', linewidth=1.5, label='PHNN')
    ax.plot(t_long, energy_baseline, 'r-', linewidth=1.5, label='Baseline NN', alpha=0.7)
    ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5, label='Training horizon')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy H(q, p)', fontsize=12)
    ax.set_title('Energy Over Time', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/phnn_hero.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUTPUT_DIR}/phnn_hero.png')


def generate_phase_portrait(phmodel, basemodel, system, t_axis):
    """Short-horizon phase portrait comparison."""
    x0 = [1.0, 0.0]
    x_exact, *_ = system.sample_trajectory(t_axis, x0=x0)
    x_phnn, _ = phmodel.simulate_trajectory(integrator=False, t_sample=t_axis, x0=x0)
    x_baseline, _ = basemodel.simulate_trajectory(integrator=False, t_sample=t_axis, x0=x0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(x_exact[:, 0], x_exact[:, 1], 'k--', linewidth=2, label='Exact')
    ax.plot(x_phnn[:, 0], x_phnn[:, 1], 'b-', linewidth=1.5, label='PHNN')
    ax.plot(x_baseline[:, 0], x_baseline[:, 1], 'r-', linewidth=1.5, label='Baseline NN')
    ax.plot(x0[0], x0[1], 'ko', markersize=8)
    ax.set_xlabel('Position (q)')
    ax.set_ylabel('Momentum (p)')
    ax.set_title('Phase Portrait (t = 0 to 10)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    ax = axes[1]
    ax.plot(t_axis, x_exact[:, 0], 'k--', linewidth=2, label='Exact')
    ax.plot(t_axis, x_phnn[:, 0], 'b-', linewidth=1.5, label='PHNN')
    ax.plot(t_axis, x_baseline[:, 0], 'r-', linewidth=1.5, label='Baseline NN')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position (q)')
    ax.set_title('Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/phase_portrait_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUTPUT_DIR}/phase_portrait_comparison.png')


def generate_training_data_viz(system, t_axis, M):
    """Training data visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for i in range(5):
        x_traj, *_ = system.sample_trajectory(t_axis)
        ax.plot(x_traj[:, 0], x_traj[:, 1], alpha=0.7)
        ax.plot(x_traj[0, 0], x_traj[0, 1], 'ko', markersize=4)
    ax.set_xlabel('Position (q)')
    ax.set_ylabel('Momentum (p)')
    ax.set_title('Training Trajectories (Phase Space)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x_traj, *_ = system.sample_trajectory(t_axis, x0=[1.0, 0.0])
    energies = [x_traj[i].T @ M @ x_traj[i] for i in range(len(t_axis))]
    ax.plot(t_axis, energies, 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy H(q, p)')
    ax.set_title('Energy Decay (Damping = 0.3)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_data.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUTPUT_DIR}/training_data.png')


def generate_damping_recovery(phmodel, system):
    """Damping constant recovery visualization."""
    learned = phmodel.R().detach().numpy()
    true_R = system.dissipation_matrix

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = ['Position (q)', 'Momentum (p)']
    true_vals = [true_R[0, 0], true_R[1, 1]]
    learned_vals = [learned[0, 0], learned[1, 1]]

    width = 0.35
    positions = np.arange(len(x))
    ax.bar(positions - width / 2, true_vals, width, label='True', color='black', alpha=0.7)
    ax.bar(positions + width / 2, learned_vals, width, label='Learned', color='#2196F3')
    ax.set_xticks(positions)
    ax.set_xticklabels(x)
    ax.set_ylabel('Damping Coefficient')
    ax.set_title('Damping Recovery: True vs Learned')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/damping_recovery.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUTPUT_DIR}/damping_recovery.png')


if __name__ == '__main__':
    print('Generating images for UC3 documentation...\n')

    print('1. Creating system and training models...')
    system, M = make_system()
    phmodel, basemodel, t_axis = train_models(system)

    print('\n2. Generating hero image (long-horizon comparison)...')
    generate_hero_image(phmodel, basemodel, system, M)

    print('\n3. Generating phase portrait comparison...')
    generate_phase_portrait(phmodel, basemodel, system, t_axis)

    print('\n4. Generating training data visualization...')
    generate_training_data_viz(system, t_axis, M)

    print('\n5. Generating damping recovery plot...')
    generate_damping_recovery(phmodel, system)

    print(f'\nDone! Images saved to {OUTPUT_DIR}/')
