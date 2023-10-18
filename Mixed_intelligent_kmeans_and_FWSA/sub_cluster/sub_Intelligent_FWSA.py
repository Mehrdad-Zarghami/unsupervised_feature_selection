''' 
Consideration for sub-clustering version of intelligent-fwsa:

{ # 1,2,3, and 5 modifications should be done in the sub_fwsa_2
  # 4 modification should be done in this file, i.e. sub_intelligent_fwsa.py }

1. Weight Initialization --> vector to matrix (n_clusters, n_features)
2. a and b for each cluster: for cluster k:
     a_v = sum( d( y_iv, c_kv ) ) for all the instances in cluster k;
         where y_iv is feature v of instance yi and c_kv is the feature v of center of cluster k
     b_v = N_k * d( C_kv, c_v ) ) ;
         where N_k is the cardinality of cluster K and c_v is the center of dataset. I think for normalized data it should be a zero vector
3. Changing the cost function
4. Save the set of weights for cluster with center t in each iteration of while loop along side with its cardinality 
5. initial the last clustering stage which is applied over all the dataset with k cluster ,
    where the the initial center and initial weights are obtained from the first k item of Z sorted based on
    the cardinality of clusters. Z is created during i_kmeans and each item of that contains center of a cluster, its cardinality and corresponding set of weights

'''

import numpy as np

from sub_fwsa_2 import sub_fwsa_2
from sub_fwsa_utils import weighted_distance
from sub_fwsa_utils import clusters_vec


# def euclidean(point, data):
#     """
#     Euclidean distance between point & data.
#     Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
#     """
#     return np.sqrt(np.sum((point - data)**2, axis=1))

def normalizer(data):
    return (data - np.mean(data,axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


def intelligent_fwsa(X, desire_number_k = 3):
    """Intelligent KMeans that actually is a KMeans with a smart initialization of centers 
    and also it determines an upper bound for the number of clusters that we can have in a dataset.

    Args:
        X (_type_): Input  data
        desire_number_k (int, optional): . Defaults to 3.

    Returns:
        _type_: It returns a fitted Kmeans object based on the input data X, the number of desired clusters,
          and also it gives us the centers that has been fined based on Intelligent KMeans.
    """

    X_copy = X.copy()

    # n_samples = X.shape[0] # Number of samples
    n_features = X.shape[1] # Number of features

    init_weights = (1/n_features) * np.ones(n_features).squeeze()# --> initial weights
    

    i_clusters = 0 # Number of clusters that we have found
    #1
    c = np.mean(X, axis=0) # c is not going to change in the loop
    Z = {}

    while True:

        #2 D is the list of Distances to the Center of all data
        D = weighted_distance(c, X, init_weights)
        # D = euclidean(c, X)
        ind_t = np.argmax(D) # the farthest point to c
        t = X[ind_t]

        #3
        
        fwsa_k = 2
        init_centers = np.array((c,t)) # ndarray --> (2, n_features) .
        #c is the center of original dataset and t is the farthest point of remaining point to  c
        history = sub_fwsa_2(X, fwsa_k, one_center_fixed=True, initial_centers= init_centers,initial_weights=None)
        U = history["U"][-1]
        # Modification #--> From here I have to continue
        weights_for_cluster_t = history["W"][-1][1]#--> we have two set of weights, one for cluster with center "c", and one for center "t". so we keep teh second set of weights 
        weights_for_cluster_c = history["W"][-1][0] 
        clusters = clusters_vec(U)
        cluster_t_indexes = list(np.where(clusters == 1)[0])
            
        #4
        theta = 1
        cluster_t_cardinality =len(cluster_t_indexes)
        if cluster_t_cardinality >= theta:
            Z[i_clusters] = (t, cluster_t_cardinality, weights_for_cluster_t)
            i_clusters += 1 

        #5 Residual indexes, after removing the cluster_t_indexes
        res_indexes = list(set(np.arange(X.shape[0])) - set(cluster_t_indexes))
        X = X[res_indexes]

        #6
        if len(res_indexes) and len(cluster_t_indexes) > 0: # The and part is added because There maybe situation that the residual points all go to cluster with c center and t cluster be empty 
            continue
        else:
            Z[i_clusters] = (X, len(res_indexes), weights_for_cluster_c)
            break
    #6.5

    deducted_z = [v for k, v, wight in Z.items()]
    deducted_z = sorted(deducted_z, key=lambda x: x[-1], reverse = True)
    # deducted_z = sorted(deducted_z, key=lambda x: x[-2], reverse = True)
    deducted_z = deducted_z[:desire_number_k]
    deducted_z = [zz[0] for zz in deducted_z]
    deducted_z = np.array(deducted_z)
        
    #7

    # Specify the number of clusters (K)
    k = len(deducted_z)
    # init_centers = [v[0] for k,v in Z.items()]

    # # Create the KMeans object
    # kmeans = KMeans(n_clusters=desire_number_k, init=deducted_z)

    # # Fit the model to the data
    # fit_model = kmeans.fit(X_copy)
    history = sub_fwsa_2(X_copy, desire_number_k, one_center_fixed=False, initial_centers= deducted_z, )

    return history, deducted_z




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_iris
    from sklearn.metrics import adjusted_rand_score
    from sub_fwsa_utils import clusters_vec
    # Load the Iris dataset
    iris = load_iris()

    # Access the features (X) and target (y) data
    X = iris.data
    y = iris.target

    import numpy as np
    def normalizer(data):
        return (data - np.mean(data,axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    X = normalizer(X)


    history, deducted_z = intelligent_fwsa(X, 3)
    U = history['U'][-1]
    clusters = clusters_vec(U)
    a_r_s = adjusted_rand_score(y, clusters )
    # Get the cluster centers and the labels for each data point  
    print(f'Centers:\n  {deducted_z}') 
    print(a_r_s)
    


    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # # Subplot 1: K-means Clustering
    # sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=labels, palette='tab10', ax=axes[0])
    # axes[0].scatter(cluster_centers[:, 2], cluster_centers[:, 3], marker='X', color='red', s=200)
    # axes[0].scatter(deducted_z[:, 2], deducted_z[:, 3], marker='o', color='black', s=50)
    # axes[0].set_xlabel('Petal Length (cm)')
    # axes[0].set_ylabel('Petal Width (cm)')
    # axes[0].set_title('K-means Clustering of Iris Dataset')

    # # Subplot 2: True Classes
    # sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y, palette='tab10', ax=axes[1])
    # axes[1].scatter(cluster_centers[:, 2], cluster_centers[:, 3], marker='X', color='red', s=200)
    # axes[1].scatter(deducted_z[:, 2], deducted_z[:, 3], marker='o', color='black', s=50)
    # axes[1].set_xlabel('Petal Length (cm)')
    # axes[1].set_ylabel('Petal Width (cm)')
    # axes[1].set_title('True Classes of Iris Dataset')

    # # Adjust layout and display the plot
    # plt.tight_layout()
    # plt.show()