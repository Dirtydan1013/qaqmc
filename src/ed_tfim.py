"""
Exact Diagonalization for Transverse Field Ising Model (TFIM)
H = -J Σ(sz_i sz_j) - h Σ(sx_i)

For verification of SSE results on small systems.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import scipy.linalg as la


class ED_TFIM:
    def __init__(self, N, J=1.0, h=1.0):
        """
        N: Number of sites (periodic chain)
        J: Ising coupling
        h: Transverse field strength
        """
        self.N = N
        self.J = J
        self.h = h
        self.dim = 2**N
        
        # Build Hamiltonian
        self.H = self._build_hamiltonian()
        
        # Cache for eigenvalues/eigenvectors
        self._eigenvalues = None
        self._eigenvectors = None
    
    def _build_hamiltonian(self):
        """Build the full Hamiltonian matrix in the sz basis"""
        N = self.N
        dim = self.dim
        
        # Use sparse matrix for efficiency
        H = sparse.lil_matrix((dim, dim), dtype=np.float64)
        
        for state in range(dim):
            # Extract spins: state = Σ s_i * 2^i, s_i ∈ {0, 1}
            # Map to sz: sz_i = 2*s_i - 1 ∈ {-1, +1}
            spins = [(state >> i) & 1 for i in range(N)]
            sz = [2 * s - 1 for s in spins]
            
            # Ising term: -J Σ sz_i sz_{i+1}
            for i in range(N):
                j = (i + 1) % N
                H[state, state] += -self.J * sz[i] * sz[j]
            
            # Transverse field term: -h Σ sx_i
            # sx flips spin i: |...0_i...> <-> |...1_i...>
            for i in range(N):
                # Flip bit i
                flipped_state = state ^ (1 << i)
                H[state, flipped_state] += -self.h
        
        return H.tocsr()
    
    def diagonalize(self, full=False):
        """Compute eigenvalues and eigenvectors"""
        if self._eigenvalues is not None:
            return self._eigenvalues, self._eigenvectors
        
        if full or self.dim <= 64:
            # Full diagonalization for small systems
            H_dense = self.H.toarray()
            self._eigenvalues, self._eigenvectors = la.eigh(H_dense)
        else:
            # Sparse diagonalization - get lowest eigenvalues
            k = min(50, self.dim - 2)
            self._eigenvalues, self._eigenvectors = eigsh(
                self.H, k=k, which='SA'
            )
            # Sort by eigenvalue
            idx = np.argsort(self._eigenvalues)
            self._eigenvalues = self._eigenvalues[idx]
            self._eigenvectors = self._eigenvectors[:, idx]
        
        return self._eigenvalues, self._eigenvectors
    
    def partition_function(self, beta):
        """Compute partition function Z = Tr[exp(-beta*H)]"""
        eigenvalues, _ = self.diagonalize()
        # Shift to avoid overflow
        E_min = eigenvalues[0]
        Z = np.sum(np.exp(-beta * (eigenvalues - E_min)))
        return Z, E_min
    
    def thermal_energy(self, beta):
        """Compute thermal average of energy <E> = Tr[H exp(-beta*H)] / Z"""
        eigenvalues, _ = self.diagonalize()
        E_min = eigenvalues[0]
        
        # Boltzmann weights
        weights = np.exp(-beta * (eigenvalues - E_min))
        Z = np.sum(weights)
        
        # <E> = Σ E_n * exp(-beta*E_n) / Z
        E_avg = np.sum(eigenvalues * weights) / Z
        
        return E_avg
    
    def thermal_magnetization_z(self, beta):
        """Compute thermal average of |<mz>| = |<Σ sz_i / N>|"""
        eigenvalues, eigenvectors = self.diagonalize()
        E_min = eigenvalues[0]
        N = self.N
        
        # Build mz operator
        mz_diag = np.zeros(self.dim)
        for state in range(self.dim):
            spins = [(state >> i) & 1 for i in range(N)]
            sz = [2 * s - 1 for s in spins]
            mz_diag[state] = np.sum(sz) / N
        
        # Boltzmann weights
        weights = np.exp(-beta * (eigenvalues - E_min))
        Z = np.sum(weights)
        
        # <|mz|> = Σ_n |<n|mz|n>| * exp(-beta*E_n) / Z
        mz_avg = 0.0
        for n, (E_n, psi_n) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # <n|mz|n> = Σ_s |psi_n(s)|^2 * mz(s)
            mz_n = np.sum(np.abs(psi_n)**2 * mz_diag)
            mz_avg += abs(mz_n) * weights[n]
        mz_avg /= Z
        
        return mz_avg
    
    def thermal_magnetization_x(self, beta):
        """Compute thermal average of <mx> = <Σ sx_i / N>"""
        eigenvalues, eigenvectors = self.diagonalize()
        E_min = eigenvalues[0]
        N = self.N
        
        # Boltzmann weights
        weights = np.exp(-beta * (eigenvalues - E_min))
        Z = np.sum(weights)
        
        # Build sx operator action and compute expectation
        mx_avg = 0.0
        for n, (E_n, psi_n) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # <n|mx|n> = (1/N) Σ_i <n|sx_i|n>
            mx_n = 0.0
            for i in range(N):
                # sx_i flips bit i
                for state in range(self.dim):
                    flipped = state ^ (1 << i)
                    mx_n += np.conj(psi_n[state]) * psi_n[flipped]
            mx_n /= N
            mx_avg += np.real(mx_n) * weights[n]
        mx_avg /= Z
        
        return mx_avg
    
    def ground_state_energy(self):
        """Get ground state energy"""
        eigenvalues, _ = self.diagonalize()
        return eigenvalues[0]
    
    def all_thermal_observables(self, beta):
        """Compute all thermal observables at once"""
        return {
            'energy': self.thermal_energy(beta),
            'magnetization_z': self.thermal_magnetization_z(beta),
            'magnetization_x': self.thermal_magnetization_x(beta),
            'ground_state_energy': self.ground_state_energy()
        }


if __name__ == "__main__":
    # Test ED
    ed = ED_TFIM(N=4, J=1.0, h=1.0)
    
    print("Testing ED for N=4, J=1.0, h=1.0")
    print(f"Ground state energy: {ed.ground_state_energy():.6f}")
    
    for beta in [1.0, 2.0, 4.0]:
        obs = ed.all_thermal_observables(beta)
        print(f"\nbeta = {beta}:")
        print(f"  Energy: {obs['energy']:.6f}")
        print(f"  |mz|: {obs['magnetization_z']:.6f}")
        print(f"  mx: {obs['magnetization_x']:.6f}")
