import numpy as np
import math
import matplotlib.pyplot as plt
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

    # Set up base figure: The contour map
    # fig, ax = plt.subplots(figsize=(8,6))
    # fig.set_tight_layout(True)

    # img = ax.imshow(z, extent=[a[0],a[1],a[0],a[1]], origin='lower', cmap='viridis', alpha=0.5)
    # fig.colorbar(img, ax=ax)
    # ax.plot([x_min], [y_min], marker='x', markersize=50, color="red")
    # contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
    # ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    # pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
    # p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
    # p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
    # gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
    # ax.set_xlim([a[0],a[1]])
    # ax.set_ylim([a[0],a[1]])


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

    # def animate(i):
    #     "Steps of PSO: algorithm update and show in plot"
    #     title = 'Iteration {:02d}'.format(i)
    #     # Update params
    #     update()
    #     apply_boundaries(X, V, a[0], a[1])
    #     # Set picture
    #     ax.set_title(title)
    #     pbest_plot.set_offsets(pbest.T)
    #     p_plot.set_offsets(X.T)
    #     p_arrow.set_offsets(X.T)
    #     p_arrow.set_UVC(V[0], V[1])
    #     gbest_plot.set_offsets(gbest.reshape(1,-1))
    #     return ax, pbest_plot, p_plot, p_arrow, gbest_plot


    # Set iteration of update function and frame interval

    iteration = list(range(1,iter))
    for _ in iteration:
        V, X, pbest, pbest_obj, gbest, gbest_obj = update(V, X, pbest, pbest_obj, gbest, gbest_obj)
        apply_boundaries(X, V, a[0], a[1])
    # from matplotlib.animation import FuncAnimation
    # anim = FuncAnimation(fig, animate, frames=iteration, interval=100, blit=False, repeat=False)
    # anim
    # from IPython.display import HTML
    # HTML(anim.to_html5_video())
    return gbest_list