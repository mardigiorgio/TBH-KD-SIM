"""
Simple Pendulum Simulation using NVIDIA Warp
Simulates 5 pendulums in parallel using explicit Euler integration
"""

import time
import warp as wp
import numpy as np

wp.init()

NUM_PENDULUMS = 10000
GRAVITY = 9.81
LENGTH = 1.0
DT = 0.001
NUM_STEPS = 1000

def initialize_angles(num_pendulums: int):
    """Initialize pendulum angles with small variations"""
    base_angle = 1.0  # radians
    angles = np.array([base_angle + i * 0.01 for i in range(num_pendulums)], dtype=np.float32)
    print(f"Initial angles (rad): {angles}")
    return angles


@wp.kernel
def euler_step(
    theta: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    omega: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    length: wp.float32,
    gravity: wp.float32,
    dt: wp.float32,
):
    """Explicit Euler integration for pendulum dynamics"""
    tid = wp.tid()

    th = theta[tid]
    om = omega[tid]

    alpha = -(gravity / length) * wp.sin(th)

    omega[tid] = om + alpha * dt
    theta[tid] = th + om * dt


def run_simulation():
    initial_angles = initialize_angles(NUM_PENDULUMS)
    initial_velocities = np.zeros(NUM_PENDULUMS, dtype=np.float32)

    theta = wp.array(initial_angles, dtype=float)
    omega = wp.array(initial_velocities, dtype=float)

    print(f"Simulating {NUM_PENDULUMS} pendulums in parallel")
    print(f"Time step: {DT}s, Steps: {NUM_STEPS}")
    print("-" * 50)

    start = time.perf_counter()

    for step in range(NUM_STEPS):
        wp.launch(
            kernel=euler_step,
            dim=NUM_PENDULUMS,
            inputs=[theta, omega, LENGTH, GRAVITY, DT],
        )

    wp.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Final angles (rad): {theta.numpy()}")
    print(f"Final velocities (rad/s): {omega.numpy()}")
    print("-" * 50)
    print(f"Elapsed time: {elapsed:.4f}s")


if __name__ == "__main__":
    run_simulation()
