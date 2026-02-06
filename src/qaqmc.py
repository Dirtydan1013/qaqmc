"""
Main QAQMC Class
Orchestrates the simulation, data structures, and high-level logic.
"""

import numpy as np
from .updates import diagonal_update_qaqmc, cluster_update_qaqmc
from .measurements import measure_observables

class QAQMC:
    def __init__(self, L, sf=0.6, c=0.5):
        self.L = L
        self.sf = sf
        self.c = c
        
        # Operator list length M scales with system size (Critical Scaling: M ~ L^3)
        self.M = max(int(L**3 / c), 100)
        
        # Linear ramp parameter s(t)
        self.s_values = np.linspace(1e-6, sf, self.M)
        
        # Initialize Operators
        self.op_type = np.zeros(self.M, dtype=np.int8)
        self.op_site1 = np.zeros(self.M, dtype=np.int32)
        self.op_site2 = np.full(self.M, -1, dtype=np.int32)
        
        # Initial spins state |Ψ0⟩ (Random Z-basis Sample of |+x⟩)
        self.spins = np.random.randint(0, 2, size=L, dtype=np.int32)
        
        # Initialize Operators with valid random configuration based on weights
        for p in range(self.M):
            s_val = self.s_values[p]
            w_f = 1 - s_val  # Type 1: Flip
            w_c = 1 - s_val  # Type 2: Const
            w_b = 2 * s_val  # Type 3: Bond
            w_tot = w_f + w_c + w_b
            
            r = np.random.random() * w_tot
            if r < w_f:
                self.op_type[p] = 1
                self.op_site1[p] = np.random.randint(0, L)
            elif r < w_f + w_c:
                self.op_type[p] = 2
                self.op_site1[p] = np.random.randint(0, L)
            else:
                self.op_type[p] = 3
                i = np.random.randint(0, L)
                self.op_site1[p] = i
                self.op_site2[p] = (i+1)%L

    def mc_step(self):
        """Perform one Monte Carlo sweep (Diagonal + Cluster Update)."""
        diagonal_update_qaqmc(self.op_type, self.op_site1, self.op_site2, self.s_values, self.spins, self.L, self.M)
        cluster_update_qaqmc(self.op_type, self.op_site1, self.op_site2, self.spins, self.L, self.M)
        
    def run(self, n_therm=500, n_measure=1000, n_skip=5, n_s_points=50):
        """
        Run the QAQMC simulation.
        
        Args:
            n_therm: Thermalization steps.
            n_measure: Number of measurement steps.
            n_skip: MC steps to skip between measurements.
            n_s_points: Number of s points to sample along the ramp.
        """
        # Thermalization
        for _ in range(n_therm):
            self.mc_step()
            
        # Determine measurement time points
        s_targets = np.linspace(0.01, self.sf, n_s_points)
        t_targets = np.floor(s_targets * self.M / self.sf).astype(np.int32)
        t_targets = np.clip(t_targets, 0, self.M)
        t_targets = np.sort(np.unique(t_targets))
        
        # Map t back to s for result dictionary
        real_s = [] 
        for t in t_targets:
            if t == 0: real_s.append(0)
            else: real_s.append(self.s_values[t-1])
        real_s = np.array(real_s)

        mz2_acc = np.zeros(len(t_targets))
        mz4_acc = np.zeros(len(t_targets))
        abs_acc = np.zeros(len(t_targets))
        
        # Measurement Loop
        for _ in range(n_measure):
            for _ in range(n_skip):
                self.mc_step()
                
            m2, m4, ma = measure_observables(self.op_type, self.op_site1, self.spins, t_targets, self.L)
            mz2_acc += m2
            mz4_acc += m4
            abs_acc += ma
            
        # Process Results
        results = {}
        for i, t in enumerate(t_targets):
            s = real_s[i]
            mz2 = mz2_acc[i] / n_measure
            mz4 = mz4_acc[i] / n_measure
            absmz = abs_acc[i] / n_measure
            
            chi = self.L * (mz2 - absmz**2)
            U = 1.5 * (1 - mz4 / (3 * mz2**2)) if mz2 > 1e-12 else 0.0
            
            results[s] = {'chi': chi, 'U': U, 's': s}
            
        return results
