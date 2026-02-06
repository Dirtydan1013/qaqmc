"""
Visualization Utils
Responsible for generating plots and reproduction figures.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .qaqmc import QAQMC

def reproduce_figure2(Ls=[8, 16, 32], sf=0.6, n_measure=2000):
    """Reproduce Figure 2 from the paper (Susceptibility and Binder Cumulant)."""
    print("Reproducing Figure 2...")
    plt.figure(figsize=(8, 10))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    for L in Ls:
        print(f"  Simulating L={L}...")
        # Constant c approx 4^3 / 240 based on paper
        q = QAQMC(L, sf=sf, c=0.26) 
        res = q.run(n_therm=1000, n_measure=n_measure, n_skip=4)
        
        s_vals = sorted(res.keys())
        chi = [res[s]['chi'] for s in s_vals]
        U = [res[s]['U'] for s in s_vals]
        
        ax1.plot(s_vals, chi, 'o-', label=f'L={L}', markersize=3)
        ax2.plot(s_vals, U, 'o-', label=f'L={L}', markersize=3)

    ax1.set_ylabel(r'$\chi$')
    ax1.set_xlabel('s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('U')
    ax2.set_xlabel('s')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure2_qaqmc.png')
    print("Saved figure2_qaqmc.png")
