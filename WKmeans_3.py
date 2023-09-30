
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
    # initial centers randomly by choosing from hte dataset randomly
    Z_initial_index = np.random.choice(range(n_samples), size=n_clusters,replace=False)
    # Centers of clusters of random samples of data, but they can be any random data_points not necessarily in dataset
    Z = X[Z_initial_index , : ]
    history['Z'].append(Z)
    
    # Generate random weights that sum up to 1

    weights = np.random.dirichlet(np.ones(n_features), size=1).squeeze()
    # weights = np.array([1/4, 1/4, 1/4, 1/4])
    history['W'].append(weights)

    initial_U = np.zeros((X.shape[0], Z.shape[0]))
    history['U'].append(initial_U)

    # put every thing together go for a while loop
    Z = history['Z'][-1] # the last update of Z
    weights = history['W'][-1] # the last update of W

    while True:

        # P1 --> update U
        U = u_calculation(X, Z, weights, beta)
        history['U'].append(U)
        # update cost
        c = cost_function(U, Z, weights, X, beta)
        history['cost'].append(c)
        if (history['U'][-1] == history['U'][-2]).all():
            break

        #P2 --> update Z     
        Z = update_Z(U, Z, X) # new update of Z
        history['Z'].append(Z)
        # Update cost
        c = cost_function(U, Z, weights, X, beta)
        history['cost'].append(c)
    
        # P3 --> update  weights
        weights = weight_update(X, U, Z, weights, beta)
        history['W'].append(weights)
        #update cost
        c = cost_function(U, Z, weights, X, beta)
        history['cost'].append(c)

    return history


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import adjusted_rand_score
    import matplotlib.pyplot as plt
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
    repetition = 20 # number of repetition for each beta
    beta_values = np.arange(1.1, 3.1, 0.1).round(1).tolist()


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
        # print(average_adjusted_rand_score_array)
        # print(history["cost"])

    # Plot the data
    plt.plot(beta_values,average_adjusted_rand_score_array )

    # Add labels and title
    plt.xlabel('beta values')
    plt.ylabel(f'Average of Adjusted Rand Score for {repetition} repetition of Beta')
    plt.title('First Algorithm ')

    # Show the plot
    plt.show()
