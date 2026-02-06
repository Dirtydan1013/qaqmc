"""
QAQMC Measurement Modules (Numba)
"""

import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def measure_observables(op_type, op_site1, spins_init, t_points, L):
    """
    Measure observables (Magnetization and Powers) at specific time points t.
    Propagates the state through the operator list to reach each measurement point.
    """
    n_pts = len(t_points)
    mz2_vals = np.zeros(n_pts)
    mz4_vals = np.zeros(n_pts)
    abs_vals = np.zeros(n_pts)
    
    current_spins = spins_init.copy()
    current_t = 0
    
    for i in range(n_pts):
        target_t = t_points[i]
        
        # Propagate from current t to target t
        # Only Type 1 (Spin-flip) affects the state
        for p in range(current_t, target_t):
            if op_type[p] == 1:
                current_spins[op_site1[p]] = 1 - current_spins[op_site1[p]]
        
        current_t = target_t
        
        # Measure Magnetization
        mz = 0.0
        for site in range(L):
            mz += 2 * current_spins[site] - 1
        mz /= L
        
        mz2_vals[i] = mz**2
        mz4_vals[i] = mz**4
        abs_vals[i] = abs(mz)
        
    return mz2_vals, mz4_vals, abs_vals
