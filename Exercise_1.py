""" 
Mandatory assignment 1
Exercise 1
General Boris mover algorithm.
"""

from math import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()
plt.style.use("ggplot")

""" 
Inital values
"""

q = 1   # Charge
m = 1   # Mass

dt = 1e-2
k = 1000 # Time duration

# Start inital
v_0 = np.array([0, 1, 0])
x_0 = np.array([0, 1, 0])

E = np.array([1, 0, 0])  # E-field (vec)  
B = np.array([0, 0, 0])  # Magnetic field (vec)

# Creating a new array of given shape, filled with zeros. 
v = np.zeros((k, 3))  
x = np.zeros((k, 3))

""" 
Main
"""

for time in range(k):
    t = q / m * (dt/2) * B 
    s = 2 * t / (1 + t*t)
    
    v_minus = v + q / m * (dt/2) * E
    v_prime = v_minus + np.cross(v_minus,t)
    v_plus = v_minus + np.cross(v_prime, s)
    
    v = v_plus + q / m * (dt / 2) * E # Finding a new velocity [v]
    x += v * dt     # Updating posisjon with time and velocity
    
    x[time, :] = x_0
    v[time, :] = v_0

""" 
Plot
"""

plt.rc(['font.sans-serif'])
plt.plot(x[:, 0], x[:, 1], color = "teal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("External fields: B=1,E=0")
plt.show()

# print 100 values with numpy
