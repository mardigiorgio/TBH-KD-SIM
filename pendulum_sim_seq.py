"""
Simple Pendulum Simulation - Sequential Version
Simulates pendulums sequentially using explicit Euler integration
"""

import time
import numpy as np

NUM_PENDULUMS = 10000
GRAVITY = 9.81
LENGTH = 1.0
DT = 0.001
NUM_STEPS = 1000


def initialize_angles(num_pendulums: int):
    """Initialize pendulum angles with small variations"""
    base_angle = 1.0  # radians
    angles = np.array([base_angle + i * 0.01 for i in range(num_pendulums)], dtype=np.float32)
    return angles


def euler_step(theta: np.ndarray, omega: np.ndarray, length: float, gravity: float, dt: float):
    """Explicit Euler integration for pendulum dynamics - sequential"""
    for i in range(len(theta)):
        alpha = -(gravity / length) * np.sin(theta[i])
        omega[i] = omega[i] + alpha * dt
        theta[i] = theta[i] + omega[i] * dt


def run_simulation():
    initial_angles = initialize_angles(NUM_PENDULUMS)
    initial_velocities = np.zeros(NUM_PENDULUMS, dtype=np.float32)

    theta = initial_angles.copy()
    omega = initial_velocities.copy()

    print(f"Simulating {NUM_PENDULUMS} pendulums sequentially")
    print(f"Time step: {DT}s, Steps: {NUM_STEPS}")
    print("-" * 50)

    start = time.perf_counter()

    for step in range(NUM_STEPS):
        euler_step(theta, omega, LENGTH, GRAVITY, DT)

    elapsed = time.perf_counter() - start

    print(f"Final angles (rad): {theta}")
    print(f"Final velocities (rad/s): {omega}")
    print("-" * 50)
    print(f"Elapsed time: {elapsed:.4f}s")


if __name__ == "__main__":
    run_simulation()
