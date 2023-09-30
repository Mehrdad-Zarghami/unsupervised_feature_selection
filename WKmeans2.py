
import numpy as np

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
    Z_initial_index = np.random.choice(range(n_samples), size=n_clusters)
    # Centers of clusters of random samples of data, but they can be any random data_points not necessarily in dataset
    Z = X[Z_initial_index , : ]
    history['Z'].append(Z)

    # Generate random weights that sum up to 1
    weights = np.random.dirichlet(np.ones(n_features), size=1).squeeze()
    history['W'].append(weights)

    #Calculating  U
    U = u_calculation(X, Z, weights)
    history['U'].append(U)

    #Calculating cost function
    c_t = cost_function(U, Z, weights, X, beta)
    history['cost'].append(c_t)

    Z_t = update_Z(U, Z, X) # new update of Z
    history['Z'].append(Z_t)

    # Update cost
    c_t = cost_function(U, Z_t, weights, X, beta) 
    history['cost'].append(c_t)

    # weight update
    weights_t = weight_update(X, U, Z, weights, beta=2)
    history['W'].append(weights_t)

    c_t = cost_function(U, Z, weights_t, X, beta = 2)
    history['cost'].append(c_t)



    # put every thing together go for a while loop


    while True:


        # P1 --> update U
        Z = history['Z'][-1] # the last update of Z
        weights = history['W'][-1] # the last update of W
        U_t = u_calculation(X, Z, weights)
        history['U'].append(U_t)
        if (history['U'][-1] == history['U'][-2]).all():
            break
        history['U'].append(U_t)
        # update cost
        c_t = cost_function(U_t, Z, weights, X, beta)
        history['cost'].append(c_t)



        #P2 --> update Z
    

        U = history['U'][-1] # the last update of U
        Z = history['Z'][-1] # the last update of Z
        weights = history['W'][-1] # the last update of weights

        Z_t = update_Z(U, Z, X) # new update of Z
        history['Z'].append(Z_t)
        if (history["z"][-1] == history["z"][-2]).all():
            break
        history['Z'].append(Z_t)
        # Update cost
        c_t = cost_function(U, Z_t, weights, X, beta) 
        history['cost'].append(c_t)



        # P3 --> update  weights

        U = history['U'][-1] # the lsat update of U
        Z = history['Z'][-1] # the lsat update of Z
        weights_t = weight_update(X, U, Z, weights, beta)
        history['W'].append(weights_t)
        if (history["W"][-1] == history["W"][-2]).all():
            break
        history['W'].append(weights_t)
        #update cost
        c_t = cost_function(U, Z, weights_t, X, beta)
        history['cost'].append(c_t)


    
    return history

if __name__ == "__main__":

    # producing clustering dataset
    from sklearn.datasets import make_blobs

    n_samples = 200
    n_features = 6
    n_clusters = 3
    data, y = make_blobs(n_samples=n_samples, n_features=n_features , centers=n_clusters, random_state=42)

    hist = wkmeans(data, n_clusters, beta=2)
    print(hist['W'][-1])
    print(np.sum(hist['W'][-1]))