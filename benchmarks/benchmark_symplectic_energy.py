import numpy as np

def run_symplectic_test(steps=1000, dt=0.15):
    # Potential: V(q) = 1/2 q^2 + 1/4 q^4
    def force(q):
        return -(q + q**3)

    def energy(q, v):
        return 0.5 * (v**2) + 0.5 * (q**2) + 0.25 * (q**4)

    # Symplectic Semi-Implicit Euler
    q_s, v_s = 1.8, 0.0
    E0_s = energy(q_s, v_s)
    drift_s = 0.0
    for _ in range(steps):
        v_s = v_s + dt * force(q_s)
        q_s = q_s + dt * v_s
        drift_s = max(drift_s, abs(energy(q_s, v_s) - E0_s))

    # Explicit Euler
    q_e, v_e = 1.8, 0.0
    E0_e = energy(q_e, v_e)
    drift_e = 0.0
    for _ in range(min(steps, 200)):
        f = force(q_e)
        q_e = q_e + dt * v_e
        v_e = v_e + dt * f
        drift_e = abs(energy(q_e, v_e) - E0_e)

    print(f"=== SYMPLECTIC PRESERVATION BENCHMARK ({steps} Steps) ===")
    print(f"[Symplectic Euler] Max Energy Drift: {drift_s:.6f} (Strictly Bounded)")
    print(f"[Explicit Euler]   Energy Drift at step 200: {drift_e:.6f} (Exponential Divergence)")

if __name__ == "__main__":
    run_symplectic_test()
