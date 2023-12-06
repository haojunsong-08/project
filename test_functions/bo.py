# Bayesian Optimization algorithm
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

def bo(fx, iter):

    # Define the search space
    space  = [Real(-10, 10, name='x'), Real(-10, 10, name='y')]

    # Define the objective function to be minimized
    @use_named_args(space)
    def objective(**params):
        return fx(params['x'], params['y'])

    # Perform Bayesian Optimization
    result = gp_minimize(objective, space, n_calls=iter,  random_state=0)

    # # Best found parameters and function value
    # print("Best parameters: {}".format(result.x))
    # print("Best function value: {}".format(result.fun))

    return result.fun

