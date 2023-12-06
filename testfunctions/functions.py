import numpy as np
import math

# Define a class for each benchmark function
class Ackley:
    def __call__(self, x, y):
        return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

class Rastgrin:
    def __call__(self, x, y):
        return (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y)) + 20

class Sphere:
    def __call__(self, x, y):
        return (4/3)*math.pi*x**3

class Rosenbrock:
    def __call__(self, x, y):
        return (1-x)**2 + 100*(y-x**2)**2

class Booth:
    def __call__(self, x, y):
        return (x+2*y-7)**2 + (2*x+y-5)**2

class Easom:
    def __call__(self, x, y):
        return -np.cos(x)*np.cos(y)*np.exp(-(x-np.pi)**2-(y-np.pi)**2)

# Function to get bounds for each function
def get_bounds(function):
    bounds = {
        "Ackley": (-32.768, 32.768),
        "Rastgrin": (-5.12, 5.12),
        "Sphere": (-100, 100),
        "Rosenbrock": (-5, 10),
        "Booth": (-10, 10),
        "Easom": (-100, 100)
    }
    return bounds[function]

# Refactored create_benchmark function
def create_benchmark(function):
    resolution = 10000
    function_mappings = {
        "Ackley": Ackley,
        "Rastgrin": Rastgrin,
        "Sphere": Sphere,
        "Rosenbrock": Rosenbrock,
        "Booth": Booth,
        "Easom": Easom
    }

    if function in function_mappings:
        a = get_bounds(function)
        x, y = np.meshgrid(np.linspace(a[0], a[1], resolution), np.linspace(a[0], a[1], resolution))
        fx_instance = function_mappings[function]()
        z = fx_instance(x, y)
        return z, x, y, a, fx_instance
