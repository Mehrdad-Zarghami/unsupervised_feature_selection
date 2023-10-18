
import numpy as np
import random

from utils import u_calculation
from utils import cost_function
from utils import update_Z
from utils import weight_update

def wkmeans(X, k, beta=2):
    """Put data (X) in k clusters based on the weight of each feature 
    which is going to be calculated based on X and a user defined parameter beta

    Args:
        X (ndarray): input data(without label)
        k (int): the number of cluster
        beta (float):  user defined parameter that is used in the definition of the loss function

    Return:
        history (dict): a dictionary that contains "U", "Z", "W", "cost" for all time steps of updating. 
                        it is obvious that the values of last time step is important
    
    """

    n_samples = X.shape[0] # Number of samples
    n_features = X.shape[1] # Number of features
    n_clusters = k # Number of clusters

    # Dictionary of history
    history = {
        'U': [],
        'Z': [],
        'W': [],
        'cost': []

        }
    time_step = 0

    # initial centers randomly by choosing from hte dataset randomly
    Z_initial_index = np.random.choice(range(n_samples), size=n_clusters,replace=False)
    # Z_initial_index = [0, 50, 100]
    # Centers of clusters of random samples of data, but they can be any random data_points not necessarily in dataset
    Z = X[Z_initial_index , : ]
    history['Z'].append(Z)
    
    # Generate random weights that sum up to 1

    # weights = np.random.dirichlet(np.ones(n_features), size=1).squeeze()
    weights = np.array([1/4,1/4, 1/4, 1/4])
    history['W'].append(weights)




    #Initial U
    def random_U(matrix):
        for row in matrix:
            num_elements = len(row)
            if num_elements > 0:
                random_index = random.randint(0, num_elements - 1)
                for i in range(num_elements):
                    if i == random_index:
                        row[i] = 1
                    else:
                        row[i] = 0
        return matrix
    zero_U = np.zeros((X.shape[0], Z.shape[0]))
    initial_U = random_U(zero_U)
    history['U'].append(initial_U)

    # first cost 
    U = history['U'][-1]
    Z = history['Z'][-1]
    W = history['W'][-1]

    c0 = cost_function(U, Z, weights, X, beta)
    history['cost'].append(c0)







    # ################
    # # U Update
    # U = u_calculation(X, Z, weights, beta)
    # history['U'].append(U)

    # ## cost Update
    # c_t = cost_function(U, Z, weights, X, beta)
    # history['cost'].append(c_t)

    # ###############
    # # Z Update
    # Z_t = update_Z(U, Z, X) # new update of Z
    # history['Z'].append(Z_t)
    # Z = history['Z'][-1]

    # # cost Update
    # c_t = cost_function(U, Z_t, weights, X, beta) 
    # history['cost'].append(c_t)

    # ################
    # # weight update
    # weights_t = weight_update(X, U, Z, weights, beta)
    # history['W'].append(weights_t)
    # weights = history['W'][-1]

    # # cost Update
    # c_t = cost_function(U, Z, weights_t, X, beta)
    # history['cost'].append(c_t)


    # put every thing together go for a while loop


    while True:

        # P1 --> update U
        Z = history['Z'][-1] # the last update of Z
        weights = history['W'][-1] # the last update of W
        U = u_calculation(X, Z, weights, beta)
        history['U'].append(U)
        U = history['U'][-1]
        # update cost
        c_t = cost_function(U, Z, weights, X, beta)
        history['cost'].append(c_t)
        print(U)
        if (history['U'][-1] == history['U'][-2]).all():
            break

        #P2 --> update Z     
        U = history['U'][-1] # the last update of U
        Z = history['Z'][-1] # the last update of Z
        weights = history['W'][-1] # the last update of weights

        Z_t = update_Z(U, Z, X) # new update of Z
        history['Z'].append(Z_t)
        # Update cost
        c_t = cost_function(U, Z_t, weights, X, beta)
        history['cost'].append(c_t)


        # P3 --> update  weights
        U = history['U'][-1] # the lsat update of U
        Z = history['Z'][-1] # the lsat update of Z
        weights_t = weight_update(X, U, Z, weights, beta)
        history['W'].append(weights_t)
        #update cost
        c_t = cost_function(U, Z, weights_t, X, beta)
        history['cost'].append(c_t)



   

    
    return history





import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from utils import clusters_vec

# Load the Iris dataset
iris = load_iris()

# Access the features (X) and target (y) data
X = iris.data
y = iris.target

n_clusters = 3
n_features = X.shape[1]
n_samples = X.shape[0]

# Normalize Data
def normalizer(data):
    return (data - np.mean(data,axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

X_norm = normalizer(X)

# First Algorithm
repetition = 1 # number of repetition for each beta
beta_values = np.arange(1.1, 1.2, 0.1).round(1).tolist()


average_adjusted_rand_score_array = np.zeros_like(beta_values)

for b, beta in enumerate(beta_values):
    adjusted_rand_score_array = np.zeros(repetition)
    for i in range(repetition):
        history = wkmeans(X_norm, n_clusters, beta)
        U = history["U"][-1]
        clusters = clusters_vec(U)
        a_r_s = adjusted_rand_score(y, clusters )
        adjusted_rand_score_array[i] = a_r_s

    average_adjusted_rand_score_array[b] =  adjusted_rand_score_array.mean()
    print(average_adjusted_rand_score_array)
    # print(history["cost"])
