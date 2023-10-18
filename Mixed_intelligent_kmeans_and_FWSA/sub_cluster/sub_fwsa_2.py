'''
Consideration for sub-clustering version of intelligent-fwsa in the sub:
1. Weight Initialization --> vector to matrix (n_clusters, n_features)
2. a and b for each cluster: for cluster k:
     a_v = sum( d( y_iv, c_kv ) ) for all the instances in cluster k;
         where y_iv is feature v of instance yi and c_kv is the feature v of center of cluster k
     b_v = N_k * d( C_kv, c_v ) ) ;
         where N_k is the cardinality of cluster K and c_v is the center of dataset. I think for normalized data it should be a zero vector
3. Changing the cost function 
'''
import numpy as np

from sub_fwsa_utils import sub_u_calculation
from sub_fwsa_utils import sub_fwsa_cost_function
from sub_fwsa_utils import update_Z
from sub_fwsa_utils import sub_fwsa_weight_update


def sub_fwsa_2(X, k, one_center_fixed = False, initial_centers = None, initial_weights = None):
    """Put data (X) in k clusters based on the weight of each feature 
    which is going to be calculated based on X

    Args:
        X (ndarray): (n_samples, n_features) input data(without label)
        k (int): the number of cluster
        one_center_fixed (binary): If True: during Z updating, the initial center which is the center of original dataset "c" is fixed and the center of next cluster is changing 
        initial_centers (ndarray): (n_clusters, n_features)
        initial_weights (ndarray): (n_clusters, n_features)
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


    if one_center_fixed == False: # in last stages, we use fwsa for k clusters 
        center_of_dataset = np.mean(X, axis=0)
    else:    # in intermediate stages, we use fwsa for 2 clusters,(c,t) where c is initial_centers[0] and is the center of original dataset
        center_of_dataset = initial_centers[0]

    # initial centers randomly by choosing from hte dataset randomly
    if initial_centers is None:
        Z_initial_index = np.random.choice(range(n_samples), size=n_clusters,replace=False)
        # Centers of clusters of random samples of data, but they can be any random data_points not necessarily in dataset
        Z = X[Z_initial_index , : ]
    else:
        Z = initial_centers

    history['Z'].append(Z)
    
    #Initial weights
    if initial_weights is None:
        # weights = (1/n_features) * np.ones(n_features).squeeze()
        init_weight = (1/n_features) * np.ones((n_clusters, n_features)).squeeze()
    else:
        init_weight = initial_weights
  
    history['W'].append(init_weight)
    initial_U = np.zeros((X.shape[0], Z.shape[0]))
    history['U'].append(initial_U)

    # put every thing together to go for a while loop
    Z = history['Z'][-1] # the last update of Z
    # weights = history['W'][-1] # the last update of W
    weights = init_weight # the last update of W

    while True:

        # P1 --> update U
        U = sub_u_calculation(X, Z, weights)
        history['U'].append(U)
        # update cost
        cst = sub_fwsa_cost_function(U, Z, weights, X)
        history['cost'].append(cst)
        if (history['U'][-1] == history['U'][-2]).all():
            break

        #P2 --> update Z     
        Z = update_Z(U, Z, X, one_center_fixed = one_center_fixed ,center_of_original_dataset = center_of_dataset) 
        # new update of Z --> initial_centers[0] is c or center of original dataset
        history['Z'].append(Z)
        # Update cost
        cst = sub_fwsa_cost_function(U, Z, weights, X)
        history['cost'].append(cst)
    
        # P3 --> update  weights
        weights = sub_fwsa_weight_update(X, U, Z, weights)
        history['W'].append(weights)
        #update cost
        cst = sub_fwsa_cost_function(U, Z, weights, X)
        history['cost'].append(cst)

    return history


if __name__ == '__main__':
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.metrics import adjusted_rand_score
    from sub_fwsa_utils import clusters_vec

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

    no_repetition = 1

    average_adjusted_rand_score_array = np.zeros(no_repetition)
    for i in range(no_repetition):
        history = sub_fwsa_2(X_norm, n_clusters,one_center_fixed = False, initial_centers = None)
        U = history["U"][-1]
        clusters = clusters_vec(U)
        a_r_s = adjusted_rand_score(y, clusters )
        average_adjusted_rand_score_array[i] = a_r_s
        print(f"ars = {a_r_s}\n")
        print(f"Weights = ")
        print( history["W"][-1])
        print(np.sum(history["W"][-1], axis=1))
        print(f"\nCost:\n {history['cost']}\n")
        print('*************')

        
    print(average_adjusted_rand_score_array.mean())