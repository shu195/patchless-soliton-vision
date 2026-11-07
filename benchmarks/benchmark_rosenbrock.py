import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def rosenbrock(x, y):
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def grad_rosenbrock(x, y):
    dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    dy = 200.0 * (y - x**2)
    return np.array([dx, dy], dtype=np.float64)

def run_benchmark(steps=1500, lr=0.001, x0=-1.5, y0=2.0):
    print(f"=== ROSENBROCK RAVINE BENCHMARK ({steps} Steps from ({x0}, {y0})) ===")
    
    # SGD
    pos_sgd = np.array([x0, y0], dtype=np.float64)
    for _ in range(steps):
        g = np.clip(grad_rosenbrock(pos_sgd[0], pos_sgd[1]), -1e4, 1e4)
        pos_sgd -= lr * g
    loss_sgd = rosenbrock(pos_sgd[0], pos_sgd[1])
    print(f"[Standard SGD]           Final Pos: ({pos_sgd[0]:.6f}, {pos_sgd[1]:.6f}) | Loss: {loss_sgd:.6e}")

    # Heavy Ball
    pos_hb = np.array([x0, y0], dtype=np.float64)
    vel_hb = np.zeros(2, dtype=np.float64)
    dt_hb = lr * 20.0
    gamma_hb = 0.8
    for _ in range(steps):
        g = np.clip(grad_rosenbrock(pos_hb[0], pos_hb[1]), -1e4, 1e4)
        vel_hb = (1.0 - gamma_hb * dt_hb) * vel_hb - dt_hb * g
        pos_hb += dt_hb * vel_hb
    loss_hb = rosenbrock(pos_hb[0], pos_hb[1])
    print(f"[Heavy Ball E-L]         Final Pos: ({pos_hb[0]:.6f}, {pos_hb[1]:.6f}) | Loss: {loss_hb:.6e}")

    # Relativistic
    pos_rel = np.array([x0, y0], dtype=np.float64)
    p_rel = np.zeros(2, dtype=np.float64)
    c_speed = 5.0
    dt_rel = lr * 20.0
    gamma_rel = 1.2
    for _ in range(steps):
        g = np.clip(grad_rosenbrock(pos_rel[0], pos_rel[1]), -1e4, 1e4)
        norm_p = np.linalg.norm(p_rel)
        v = p_rel / np.sqrt(1.0 + (norm_p / c_speed)**2)
        p_rel = p_rel - dt_rel * (g + gamma_rel * v)
        norm_p_new = np.linalg.norm(p_rel)
        v_next = p_rel / np.sqrt(1.0 + (norm_p_new / c_speed)**2)
        pos_rel += dt_rel * v_next
    loss_rel = rosenbrock(pos_rel[0], pos_rel[1])
    print(f"[Relativistic E-L]       Final Pos: ({pos_rel[0]:.6f}, {pos_rel[1]:.6f}) | Loss: {loss_rel:.6e}")

if __name__ == "__main__":
    run_benchmark()
