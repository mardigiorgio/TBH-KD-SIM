import numpy as np
import matplotlib.pyplot as plt

g = 9.81
L = 1.0
dt = 0.001  
num_steps = 1000

def f(y):
    theta, omega = y
    return np.array([omega, -g/L * np.sin(theta)])

def jacobian(y):
    theta, omega = y
    return np.array([
        [0, 1],
        [-g/L * np.cos(theta), 0]
    ])

def newton_step(y):
    guess = y.copy()

    for k in range(10):
        G = guess - y - dt * f(guess)
        dG = np.eye(2) - dt * jacobian(guess)

        delta = np.linalg.solve(dG, -G)
        guess = guess + delta

        if np.linalg.norm(G) < 1e-10:
            break
    y = guess
    return y


y = np.array([np.pi/4, 0.0])

dt = 0.01
t_end = 10.0
t = 0.0

history = [y.copy()]

while t < t_end:
    y = newton_step(y)

    t += dt
    history.append(y.copy())

history = np.array(history)

plt.plot(np.arange(len(history)) * dt, history[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid()
plt.show()