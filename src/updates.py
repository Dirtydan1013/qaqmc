"""
QAQMC Optimization Kernels (Numba)
Contains the core Monte Carlo update functions:
- Diagonal Update (Type 2 <-> Type 3)
- Segment-Based Cluster Update (Type 1 <-> Type 2, Spin Flips)
"""

import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def diagonal_update_qaqmc(op_type, op_site1, op_site2, s_values, spins_init, L, M):
    """
    Diagonal Update: Sampling between Type 2 (Constant) and Type 3 (Bond).
    
    Update rule:
    - Iterate through operator string.
    - If Type 2 (Constant at site i): Try swapping to Type 3 (Bond at bond b).
    - If Type 3 (Bond at bond b): Try swapping to Type 2 (Constant at site i).
    - Selection probabilities ensure detailed balance.
    """
    # Create a working copy of spins to propagate
    spins = spins_init.copy()
    
    for p in range(M):
        s = s_values[p]
        op_t = op_type[p]
        
        # Propagate Type 1 (Spin-flip) operators
        if op_t == 1:
            site = op_site1[p]
            spins[site] = 1 - spins[site]
            continue
            
        # Metropolis update for Diagonal operators (Type 2 <-> Type 3)
        if np.random.random() < 0.5:
            if op_t == 2:
                # Try Type 2 -> Type 3
                # Pick random bond to replace current site operator
                bond_idx = np.random.randint(0, L)
                i = bond_idx
                j = (bond_idx + 1) % L
                
                # Bond operator only valid between parallel spins
                if spins[i] == spins[j]:
                    # Acceptance ratio: W(3)/W(2) * (Prob_select_2 / Prob_select_3)
                    # Ratio = (2s) / (1-s)
                    ratio = (2.0 * s) / (1.0 - s + 1e-10)
                    
                    if np.random.random() < ratio:
                        op_type[p] = 3
                        op_site1[p] = i
                        op_site2[p] = j
            
            elif op_t == 3:
                # Try Type 3 -> Type 2
                # Pick random site to replace current bond operator
                site_idx = np.random.randint(0, L)
                
                # Ratio = W(2)/W(3) = (1-s) / (2s)
                ratio = (1.0 - s) / (2.0 * s + 1e-10)
                
                if np.random.random() < ratio:
                    op_type[p] = 2
                    op_site1[p] = site_idx
                    op_site2[p] = -1

@jit(nopython=True, cache=True)
def find_root(parent, x):
    """Iterative Union-Find with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root

@jit(nopython=True, cache=True)
def cluster_update_qaqmc(op_type, op_site1, op_site2, spins, L, M):
    """
    Segment-Based Space-Time Cluster Update.
    
    This update handles:
    1. Global spin flips (ergodicity).
    2. Swapping between Type 1 (Flip) and Type 2 (Constant) operators.
       (This is the only valid way to change off-diagonal operators while maintaining
        path validity constraints).
    
    Algorithm:
    1. Segment Identification: The operator string + boundaries define continuous 
       segments of constant spin value in space-time.
       Type 1 & 2 operators act as boundaries between segments.
    2. Graph Construction: Segments are nodes. Type 3 (Bond) operators link segments 
       on adjacent sites (must remain parallel).
    3. Swendsen-Wang Cluster Flip: Connected segments flip together with p=0.5.
    4. Operator Update: If two segments separated by an operator flip differently
       (relative state change), the operator type swaps (1 <-> 2).
    """
    # Initialize segment tracking
    # current_seg maps site index -> current segment ID
    current_seg = np.arange(L, dtype=np.int32) 
    next_seg_id = L
    
    # Union-Find structure for segments
    # Max possible segments: L (initial) + M (each Type 1/2 operator starts new segment)
    parent = np.arange(M + L, dtype=np.int32)
    
    # --- Pass 1: Build Clusters ---
    for p in range(M):
        op_t = op_type[p]
        
        if op_t == 3: # Bond operator
            # Link segments on sites i and j (Constraint: must be parallel)
            i = op_site1[p]
            j = op_site2[p]
            
            root_i = find_root(parent, current_seg[i])
            root_j = find_root(parent, current_seg[j])
            
            if root_i != root_j:
                parent[root_i] = root_j
                
        elif op_t == 1 or op_t == 2: # Flip or Constant operator
            # Start new segment at site i
            i = op_site1[p]
            new_s = next_seg_id
            next_seg_id += 1
            current_seg[i] = new_s
            
            # Note: We do NOT link the old and new segments.
            # They are coupled via the operator weight (approx equal for 1 and 2).
            # Whether they flip relative to each other determines the operator type update.

    # --- Pass 2: Decide Flips ---
    num_segments = next_seg_id
    
    # Decide flip for each cluster root
    root_flip = np.random.randint(0, 2, size=num_segments).astype(np.int8)
    
    # --- Pass 3: Apply Updates ---
    
    # A. Update Operators based on segment flips
    for i in range(L): current_seg[i] = i
    curr_seg_id_counter = L
    
    for p in range(M):
        op_t = op_type[p]
        
        if op_t == 1 or op_t == 2:
            i = op_site1[p]
            
            seg_before = current_seg[i]
            seg_after = curr_seg_id_counter
            curr_seg_id_counter += 1
            current_seg[i] = seg_after
            
            # Check if relative domain wall state changed
            root_before = find_root(parent, seg_before)
            root_after = find_root(parent, seg_after)
            
            f_before = root_flip[root_before]
            f_after = root_flip[root_after]
            
            if f_before != f_after:
                # Relative flip changed -> Swap operator type
                # Type 1 <-> Type 2
                op_type[p] = 3 - op_t
                
    # B. Update Initial Spin Configuration (Boundary Condition)
    # The initial state is defined by the first set of segments (0 to L-1)
    for i in range(L):
        root = find_root(parent, i)
        if root_flip[root] == 1:
            spins[i] = 1 - spins[i]
