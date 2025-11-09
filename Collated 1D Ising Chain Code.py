import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.identity(2, dtype=complex)

# Function to construct the Hamiltonian
def construct_hamiltonian(N, J, g):
    """Constructs the Hamiltonian matrix for N spins."""
    dim = 2**N  # Hilbert space dimension
    H = np.zeros((dim, dim), dtype=complex)

    # Nearest-neighbor interaction term: -J sum σ_x_j σ_x_j+1
    for j in range(N - 1):
        term = 1
        for k in range(N):
            if k == j or k == j + 1:
                term = np.kron(term, sigma_x)
            else:
                term = np.kron(term, identity)
        H -= J * term

    # External field term: -g sum σ_z_j
    for j in range(N):
        term = 1
        for k in range(N):
            if k == j:
                term = np.kron(term, sigma_z)
            else:
                term = np.kron(term, identity)
        H -= g * term

    return H

# Function to solve the Hamiltonian
def solve_hamiltonian(N, J, g):
    """Computes the eigenvalues and eigenvectors of the Hamiltonian."""
    H = construct_hamiltonian(N, J, g)
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    print(f"Lowest 4 eigenvalues for N={N}, J={J}, g={g}:")
    print(eigenvalues[:4])

    return eigenvalues, eigenvectors

# Function to apply Pauli-X to a specific site
def apply_pauli_x(state, site, N):
    """Apply the Pauli-X operator to the given site in the state vector."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    operator = 1
    for k in range(N):
        if k == site:
            operator = np.kron(operator, sigma_x)
        else:
            operator = np.kron(operator, identity)
    return operator @ state # @ is matrix multiplication, 

# Function to compute magnetisation
def compute_magnetisation(state, N):
    """Compute magnetisation for each spin site in the given state."""
    magnetisation = []
    for j in range(N):
        term = 1
        for k in range(N):
            if k == j:
                term = np.kron(term, sigma_z)
            else:
                term = np.kron(term, identity)
        mag_j = np.real(state.conj().T @ term @ state)
        magnetisation.append(mag_j)
    return magnetisation

# Task 4: Solve ground state and magnetisation
def task_4(N, J, g):
    eigenvalues, eigenvectors = solve_hamiltonian(N, J, g)
    ground_state = eigenvectors[:, 0]
    ground_energy = eigenvalues[0]

    # Perturb the ground state
    perturbed_state = apply_pauli_x(ground_state, N // 2, N)

    # Compute magnetisation
    magnetisation = compute_magnetisation(perturbed_state, N)

    print(f"Ground state energy: {ground_energy:.4f}")
    print("Magnetisation at each site after perturbation:")
    for j, mag in enumerate(magnetisation, start=1):
        print(f"Site {j}: {mag:.4f}")

# Task 5: Time evolution and magnetisation
def task_5(N, J, g):
    H = construct_hamiltonian(N, J, g)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_state = eigenvectors[:, 0]
    perturbed_state = apply_pauli_x(ground_state, N // 2, N)
    perturbed_state = perturbed_state / np.linalg.norm(perturbed_state)

    time_steps = [0.0, 0.1, 0.5, 1.0]
    for t in time_steps:
        U_t = expm(-1j * H * t)
        psi_t = U_t @ perturbed_state
        norm = np.linalg.norm(psi_t)
        print(f"Norm of the state at t={t}: {norm:.6f}")

        magnetisation = compute_magnetisation(psi_t, N)
        print(f"Magnetisation at t={t}:")
        for j, mag in enumerate(magnetisation, start=1):
            print(f"  Site {j}: {mag:.4f}")
        print("-" * 30)

# Task 6: Magnetisation dynamics over time
def task_6(N, J, g):
    H = construct_hamiltonian(N, J, g)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_state = eigenvectors[:, 0]
    perturbed_state = apply_pauli_x(ground_state, N // 2, N)
    perturbed_state = perturbed_state / np.linalg.norm(perturbed_state)

    time_steps = np.linspace(0, 4, 41)
    magnetisation_time = []

    for t in time_steps:
        U_t = expm(-1j * H * t)
        psi_t = U_t @ perturbed_state
        magnetisation = compute_magnetisation(psi_t, N)
        magnetisation_time.append(magnetisation)

    magnetisation_time = np.array(magnetisation_time)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        magnetisation_time,
        extent=[1, N, 0, 4],
        origin='lower',
        aspect='auto',
        cmap='jet'
    )
    plt.colorbar(label="Magnetisation ⟨σ_z⟩")
    plt.xlabel("Spin site")
    plt.ylabel("Time")
    plt.title("Magnetisation dynamics over time")
    plt.show()

# Main execution
if __name__ == "__main__":
    N = 7  # Number of spins
    J = 1  # Coupling constant
    g = 3  # External field

    print("Task 4:")
    task_4(N, J, g)

    print("\nTask 5:")
    task_5(N, J, g)

    print("\nTask 6:")
    task_6(N, J, g)