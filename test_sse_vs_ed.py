"""
Full comparison test: SSE vs ED for TFIM
"""

import numpy as np
from src.sse_tfim import SSE_TFIM
from src.ed_tfim import ED_TFIM


def run_test(N, beta, J, h, n_therm=1500, n_measure=3000):
    """Compare SSE and ED"""
    print(f"\n{'='*50}")
    print(f"N={N}, β={beta}, J={J}, h={h}")
    print(f"{'='*50}")
    
    # ED
    ed = ED_TFIM(N=N, J=J, h=h)
    ed_e = ed.thermal_energy(beta)
    ed_mz = ed.thermal_magnetization_z(beta)
    print(f"ED:  E = {ed_e:.4f}, |mz| = {ed_mz:.4f}")
    
    # SSE
    h_sse = max(h, 0.001)  # Avoid h=0 issues
    sse = SSE_TFIM(N=N, beta=beta, J=J, h=h_sse, L_init=max(20, int(3*beta*N)))
    r = sse.run(n_therm=n_therm, n_measure=n_measure)
    
    print(f"SSE: E_sse = {r['energy_sse']:.4f} ± {r['energy_sse_err']:.4f}")
    print(f"     E_Ising = {r['energy_ising']:.4f}")
    print(f"     |mz| = {r['magnetization']:.4f}")
    
    # Compare
    diff = abs(r['energy_sse'] - ed_e)
    rel_err = diff / max(abs(ed_e), 0.01) * 100
    print(f"Energy diff: {diff:.4f} ({rel_err:.1f}%)")
    
    pass_test = rel_err < 10  # 10% tolerance
    print(f"Result: {'✓ PASS' if pass_test else '✗ FAIL'}")
    
    return pass_test, rel_err


def main():
    np.random.seed(42)
    
    print("="*50)
    print("SSE vs ED Verification for TFIM")
    print("Based on Sandvik's algorithm (cond-mat/0303597)")
    print("="*50)
    
    tests = [
        {'N': 4, 'beta': 2.0, 'J': 1.0, 'h': 0.0},   # Pure Ising
        {'N': 4, 'beta': 1.0, 'J': 1.0, 'h': 0.5},   # Low field
        {'N': 4, 'beta': 2.0, 'J': 1.0, 'h': 1.0},   # Medium field
        {'N': 4, 'beta': 1.0, 'J': 1.0, 'h': 2.0},   # High field
        {'N': 6, 'beta': 1.5, 'J': 1.0, 'h': 1.0},   # Larger system
    ]
    
    results = []
    for params in tests:
        passed, err = run_test(**params)
        results.append((params, passed, err))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    n_passed = sum(1 for _, p, _ in results if p)
    print(f"Passed: {n_passed}/{len(results)}")
    
    for params, passed, err in results:
        status = "✓" if passed else "✗"
        print(f"  {status} N={params['N']}, β={params['beta']}, h={params['h']}: {err:.1f}% error")


if __name__ == "__main__":
    main()
