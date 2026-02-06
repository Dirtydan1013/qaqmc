"""
Main Entry Point for QAQMC Simulation
"""
import numpy as np
from src.visualization import reproduce_figure2

if __name__ == "__main__":
    np.random.seed(42)
    print("Starting QAQMC Simulation...")
    # Run simulation for L=8, 16, 32 as requested
    reproduce_figure2(Ls=[8, 16, 32], n_measure=2000)
    print("Simulation Complete.")
