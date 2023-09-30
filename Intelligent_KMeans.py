from sklearn.cluster import KMeans
import numpy as np

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))

def normalizer(data):
    return (data - np.mean(data,axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


def intelligent_kmeans(X, desire_number_k = 3):
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
    

    i_clusters = 0 # Number of clusters that we have found
    #1
    c = np.mean(X, axis=0) # c is not going to change in the loop
    Z = {}

    while True:

        #2 D is the list of Distances to the Center of all data
        D = euclidean(c, X)
        ind_t = np.argmax(D) # the farthest point to c
        t = X[ind_t]

        #3
        cluster_t_indexes = []
        while True:
            D_clusters = [(a,b) for a,b in zip(euclidean(c, X), euclidean(t, X))]

            clusters = np.argmin(D_clusters, axis=1)

            new_cluster_t_indexes = list(np.where(clusters == 1)[0])

            t = np.mean(X[new_cluster_t_indexes], axis=0)
            
            # counter += 1
            if cluster_t_indexes != new_cluster_t_indexes:
                cluster_t_indexes = new_cluster_t_indexes
                continue
            else:
                break
            
        #4
        theta = 1
        cluster_t_cardinality =len(cluster_t_indexes)
        if cluster_t_cardinality >= theta:
            Z[i_clusters] = (t, cluster_t_cardinality)
            i_clusters += 1 
        # else:
        #     print('Cardinality for Cluster {t} is less than Theta ({theta})')
        #     break # What should we do here? Remove cluster_t? 

        #5 Residual indexes, after removing the cluster_t_indexes
        res_indexes = list(set(np.arange(X.shape[0])) - set(cluster_t_indexes))
        X = X[res_indexes]

        #6
        if len(res_indexes) > 0:
            continue
        else:
            break
    #6.5

    deducted_z = [v for k, v in Z.items()]
    deducted_z = sorted(deducted_z, key=lambda x: x[-1], reverse = True)
    deducted_z = deducted_z[:desire_number_k]
    deducted_z = [zz[0] for zz in deducted_z]
    deducted_z = np.array(deducted_z)
        
    #7

    # Specify the number of clusters (K)
    k = len(deducted_z)
    # init_centers = [v[0] for k,v in Z.items()]

    # Create the KMeans object
    kmeans = KMeans(n_clusters=desire_number_k, init=deducted_z)

    # Fit the model to the data
    fit_model = kmeans.fit(X_copy)

    return fit_model, deducted_z




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import load_iris
    from sklearn.metrics import adjusted_rand_score
    # Load the Iris dataset
    iris = load_iris()

    # Access the features (X) and target (y) data
    X = iris.data
    y = iris.target

    import numpy as np
    def normalizer(data):
        return (data - np.mean(data,axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    X = normalizer(X)


    fit_ikmeans, deducted_z = intelligent_kmeans(X, 3)
    # Get the cluster centers and the labels for each data point
    cluster_centers = fit_ikmeans.cluster_centers_
    labels = fit_ikmeans.labels_
    a_r_s = adjusted_rand_score(y, labels )
    print(a_r_s)
    


    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Subplot 1: K-means Clustering
    sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=labels, palette='tab10', ax=axes[0])
    axes[0].scatter(cluster_centers[:, 2], cluster_centers[:, 3], marker='X', color='red', s=200)
    axes[0].scatter(deducted_z[:, 2], deducted_z[:, 3], marker='o', color='black', s=50)
    axes[0].set_xlabel('Petal Length (cm)')
    axes[0].set_ylabel('Petal Width (cm)')
    axes[0].set_title('K-means Clustering of Iris Dataset')

    # Subplot 2: True Classes
    sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=y, palette='tab10', ax=axes[1])
    axes[1].scatter(cluster_centers[:, 2], cluster_centers[:, 3], marker='X', color='red', s=200)
    axes[1].scatter(deducted_z[:, 2], deducted_z[:, 3], marker='o', color='black', s=50)
    axes[1].set_xlabel('Petal Length (cm)')
    axes[1].set_ylabel('Petal Width (cm)')
    axes[1].set_title('True Classes of Iris Dataset')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()





