import numpy as np

from fwsa_utils import u_calculation
from fwsa_utils import fwsa_cost_function
from fwsa_utils import update_Z
from fwsa_utils import fwsa_weight_update


def fwsa_2(X, k, one_center_fixed = False, initial_centers = None,):
    """Put data (X) in k clusters based on the weight of each feature 
    which is going to be calculated based on X

    Args:
        X (ndarray): (n_samples, n_features) input data(without label)
        k (int): the number of cluster
        initial_centers (ndarray): (n_clusters, n_features)
        one_center_fixed (binary): If True: during Z updating, the initial center which is the center of original dataset "c" is fixed and teh center of next cluster is changing 

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
    # initial centers randomly by choosing from hte dataset randomly
    if initial_centers is None:
        Z_initial_index = np.random.choice(range(n_samples), size=n_clusters,replace=False)
        # Centers of clusters of random samples of data, but they can be any random data_points not necessarily in dataset
        Z = X[Z_initial_index , : ]
    else:
        Z = initial_centers

    history['Z'].append(Z)
    
    # Generate random weights that sum up to 1

    weights = (1/n_features) * np.ones(n_features).squeeze()
    # weights = np.array([1/4, 1/4, 1/4, 1/4])
    history['W'].append(weights)
    initial_U = np.zeros((X.shape[0], Z.shape[0]))
    history['U'].append(initial_U)

    # put every thing together to go for a while loop
    Z = history['Z'][-1] # the last update of Z
    weights = history['W'][-1] # the last update of W

    while True:

        # P1 --> update U
        U = u_calculation(X, Z, weights)
        history['U'].append(U)
        # update cost
        c = fwsa_cost_function(U, Z, weights, X)
        history['cost'].append(c)
        if (history['U'][-1] == history['U'][-2]).all():
            break

        #P2 --> update Z     
        Z = update_Z(U, Z, X, one_center_fixed = one_center_fixed ,center_of_original_dataset = initial_centers[0]) 
        # new update of Z --> initial_centers[0] is c or center of original dataset
        history['Z'].append(Z)
        # Update cost
        c = fwsa_cost_function(U, Z, weights, X)
        history['cost'].append(c)
    
        # P3 --> update  weights
        weights = fwsa_weight_update(X, U, Z, weights)
        history['W'].append(weights)
        #update cost
        c = fwsa_cost_function(U, Z, weights, X)
        history['cost'].append(c)

    return history


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.metrics import adjusted_rand_score
    from fwsa_utils import clusters_vec

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

    average_adjusted_rand_score_array = np.zeros(100)
    for i in range(100):
        history = fwsa_2(X_norm, n_clusters)
        U = history["U"][-1]
        clusters = clusters_vec(U)
        a_r_s = adjusted_rand_score(y, clusters )
        average_adjusted_rand_score_array[i] = a_r_s
        
    print(average_adjusted_rand_score_array.mean())