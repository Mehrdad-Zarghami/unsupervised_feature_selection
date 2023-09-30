
import numpy as np

from entropy_utils import u_calculation
from entropy_utils import entropy_cost_function
from entropy_utils import update_Z
from entropy_utils import sub_weight_update



def ewkmeans(X, k, gama=1.5):
    """Put data (X) in k clusters based on the weight of each feature 
    which is going to be calculated based on X and a user defined parameter gama

    Args:
        X (ndarray): input data(without label)
        k (int): the number of cluster
        gama (float):  user defined parameter that is used in the definition of the loss function

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
    Z_initial_index = np.random.choice(range(n_samples), size=n_clusters,replace=False)
    # Centers of clusters of random samples of data, but they can be any random data_points not necessarily in dataset
    Z = X[Z_initial_index , : ]
    history['Z'].append(Z)
    
    # Generate random weights that sum up to 1

    weights = (1/n_features) * np.ones((n_clusters,n_features)).squeeze()
    # weights = np.array([1/4, 1/4, 1/4, 1/4])
    history['W'].append(weights)

    initial_U = np.zeros((X.shape[0], Z.shape[0]))
    history['U'].append(initial_U)

    # put every thing together go for a while loop
    Z = history['Z'][-1] # the last update of Z
    weights = history['W'][-1] # the last update of W

    while True:

        # P1 --> update U
        U = u_calculation(X, Z, weights)
        history['U'].append(U)
        # update cost
        c = entropy_cost_function(U, Z, weights, X, gama)
        history['cost'].append(c)
        if (history['U'][-1] == history['U'][-2]).all():
            break

        #P2 --> update Z     
        Z = update_Z(U, Z, X) # new update of Z
        history['Z'].append(Z)
        # Update cost
        c = entropy_cost_function(U, Z, weights, X, gama)
        history['cost'].append(c)
    
        # P3 --> update  weights
        weights = sub_weight_update(X, U, Z, weights, gama)
        history['W'].append(weights)
        #update cost
        c = entropy_cost_function(U, Z, weights, X, gama)
        history['cost'].append(c)

    return history


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import adjusted_rand_score
    import matplotlib.pyplot as plt
    from entropy_utils import clusters_vec

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


    gama = 0.3
    average_adjusted_rand_score_array = np.zeros(100)
    for i in range(100):
        history = ewkmeans(X_norm, n_clusters, gama=gama)
        U = history["U"][-1]
        clusters = clusters_vec(U)
        a_r_s = adjusted_rand_score(y, clusters )
        average_adjusted_rand_score_array[i] = a_r_s
        
    print(average_adjusted_rand_score_array.mean())

    # # First Algorithm

    # average_adjusted_rand_score_array = np.zeros_like(gama_values)

    # for b, gama in enumerate(gama_values):
    #     adjusted_rand_score_array = np.zeros(repetition)
    #     for i in range(repetition):
    #         history = ewkmeans(X_norm, n_clusters, gama)
    #         U = history["U"][-1]
    #         clusters = clusters_vec(U)
    #         a_r_s = adjusted_rand_score(y, clusters )
    #         adjusted_rand_score_array[i] = a_r_s

    #     average_adjusted_rand_score_array[b] =  adjusted_rand_score_array.mean()
    #     # print(average_adjusted_rand_score_array)
    #     # print(history["cost"])

    # # Plot the data
    # plt.plot(gama_values,average_adjusted_rand_score_array )

    # # Add labels and title
    # plt.xlabel('gama values')
    # plt.ylabel(f'Average of Adjusted Rand Score for {repetition} repetition of gama')
    # plt.title('First Algorithm ')

    # # Show the plot
    # plt.show()

