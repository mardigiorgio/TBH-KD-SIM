import numpy as np
import matplotlib.pyplot as plt

g = 9.81
L = 1.0
dt = 0.001  
num_steps = 1000

def f(y):
    theta, omega = y
    return np.array([omega, -g/L * np.sin(theta)])

y = np.array([np.pi/4, 0.0])

dt = 0.01
t_end = 10.0
t = 0.0

history = [y.copy()]

while t < t_end:
    y=y + dt * f(y)
    
    t += dt
    history.append(y.copy())

history = np.array(history)

plt.plot(np.arange(len(history)) * dt, history[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid()
plt.show()