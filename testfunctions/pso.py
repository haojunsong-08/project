import numpy as np
import math
import matplotlib.pyplot as plt
import optunity
def pso_optunity(a, fx, z, x, y, iter):
    def objective_function(x, y):
        return fx(x, y)
     # Define the search space
    search_space = {'x': [a[0], a[1]], 'y': [a[0], a[1]]}

    # Using PSO to optimize the objective function within the defined search space
    optimal_pars, details, _ = optunity.minimize(objective_function, 
                                                 num_evals=iter, 
                                                 solver_name='particle swarm', 
                                                 **search_space)
    return optimal_pars, details

    return gbest_list
def pso(a, fx, z, x, y, iter):
    # Find the global minimum
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    x_max = x.ravel()[z.argmax()]
    y_max = y.ravel()[z.argmax()] 
    #PSO algorithm
    # Hyper-parameter of the algorithm
    c1 = 0.5
    c2 = 0.5
    w = 0.6

    # Create particles
    n_particles = 100
    np.random.seed(100)
    # Values of particles
    # set the region of the particle to be the same as the region of the function
    X = np.random.uniform(a[0], a[1], (2, n_particles))
    # the velocity of the particle with random velocity sampled over a normal distribution with mean 0 and standard deviation 0.1
    V = np.random.normal(0, 0.1, (2, n_particles))

    gbest_list = []
    # Initialize data
    pbest = X
    pbest_obj = fx(X[0], X[1])
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

    def apply_boundaries(X, V, lower_bound, upper_bound):
        # Apply boundaries to positions
        np.clip(X, lower_bound, upper_bound, out=X)

        # Reverse velocity if the particle hits the boundary
        for i in range(X.shape[1]):  # loop over each particle
            for j in range(X.shape[0]):  # loop over each dimension
                if X[j, i] == lower_bound or X[j, i] == upper_bound:
                    V[j, i] *= -1

    def update(V, X, pbest, pbest_obj, gbest, gbest_obj):
        "Function to do one iteration of particle swarm optimization"
        # store the gbest for each iteration
        # Update params
        r1, r2 = np.random.rand(2)
        V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1)-X)
        X = X + V
        obj = fx(X[0], X[1])
        pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
        pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
        gbest = pbest[:, pbest_obj.argmin()]
        gbest_obj = pbest_obj.min()
        gbest_list.append(gbest_obj)
        return V, X, pbest, pbest_obj, gbest, gbest_obj

    iteration = list(range(1,iter))
    for _ in iteration:
        V, X, pbest, pbest_obj, gbest, gbest_obj = update(V, X, pbest, pbest_obj, gbest, gbest_obj)
        apply_boundaries(X, V, a[0], a[1])
    return gbest_list