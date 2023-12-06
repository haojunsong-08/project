import pandas as pd
import numpy as np
import random
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score



#data = pd.read_csv(r"D:\WFs1.csv")
#x = data.iloc[:, 1:]
#Y = data.iloc[:, 0]


from sklearn.datasets import load_iris

data = load_iris()
x = data.data
Y = data.target


scaler = StandardScaler()
X = scaler.fit_transform(x)


W = 0.5
c1 = 0.2
c2 = 0.5
n_iterations = 10
n_particles = 100


def fitness(position):

    lassifier = SVC(kernel='rbf', gamma=position[0], C=position[1])
    lassifier.fit(X, Y)

    Y_pred = cross_val_predict(lassifier, X, Y, cv=9)

    return confusion_matrix(Y, Y_pred)[0][1] + confusion_matrix(Y, Y_pred)[0][2] + confusion_matrix(Y, Y_pred)[1][0] + \
    confusion_matrix(Y, Y_pred)[1][2] + confusion_matrix(Y, Y_pred)[2][0] + confusion_matrix(Y, Y_pred)[2][1] \
        , confusion_matrix(Y, Y_pred)[0][1] + confusion_matrix(Y, Y_pred)[0][2] + confusion_matrix(Y, Y_pred)[1][0] + \
    confusion_matrix(Y, Y_pred)[1][2] + confusion_matrix(Y, Y_pred)[2][0] + confusion_matrix(Y, Y_pred)[2][1]


    #return (confusion_matrix(Y, Y_pred)[0][1] + confusion_matrix(Y, Y_pred)[0][2] + confusion_matrix(Y, Y_pred)[1][0] + \
         #  confusion_matrix(Y, Y_pred)[1][2] + confusion_matrix(Y, Y_pred)[2][0] + confusion_matrix(Y, Y_pred)[2][1]) /\
        #(confusion_matrix(Y, Y_pred)[0][1] +confusion_matrix(Y, Y_pred)[0][0]+ confusion_matrix(Y, Y_pred)[0][2] + \
        # confusion_matrix(Y, Y_pred)[1][0] + confusion_matrix(Y, Y_pred)[1][1]+confusion_matrix(Y, Y_pred)[1][2] +\
        # confusion_matrix(Y, Y_pred)[2][0] + confusion_matrix(Y, Y_pred)[2][1]+ confusion_matrix(Y, Y_pred)[2][2])

def plot(position):
    x = []
    y = []
    for i in range(0, len(particle_position)):
        x.append(particle_position[i][0])
        y.append(particle_position[i][1])
    colors = (0, 0, 0)
    plt.scatter(x, y, c = colors, alpha = 0.1)


    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.axis([0, 10, 0, 10],)
    plt.gca().set_aspect('equal', adjustable='box')
    return plt.show()


particle_position = np.array([np.array([random.random() * 10, random.random() * 10]) for _ in range(n_particles)])
pbest_position = particle_position
pbest_fitness= np.array([float('inf') for _ in range(n_particles)])
gbest_fitness= np.array([float('inf'), float('inf')])
gbest_position = np.array([float('inf'), float('inf')])
velocity = ([np.array([0, 0]) for _ in range(n_particles)])
iteration = 0
while iteration < n_iterations:
    plot(particle_position)
    for i in range(n_particles):
        fitness_cadidate = fitness(particle_position[i])
        print("error of particle-", i, "is (training, test)", fitness_cadidate, " At (gamma, c): ",
              particle_position[i])

        if (pbest_fitness[i] > fitness_cadidate[1]):
            pbest_fitness[i] = fitness_cadidate[1]
            pbest_position[i] = particle_position[i]

        if (gbest_fitness_value[1] > fitness_cadidate[1]):
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position[i]

        elif (gbest_fitness_value[1] == fitness_cadidate[1] and gbest_fitness_value[0] > fitness_cadidate[0]):
            gbest_fitness_value = fitness_cadidate
            gbest_position = particle_position[i]

    for i in range(n_particles):
        new_velocity = (W * velocity[i]) + (c1 * random.random()) * (
                    pbest_position[i] - particle_position[i]) + (c2 * random.random()) * (
                                   gbest_position - particle_position[i])
        new_position = new_velocity + particle_position[i]
        particle_position[i] = new_position

    iteration = iteration + 1


print("The best position is ", gbest_position, "in iteration number", iteration,
      fitness(gbest_position))
