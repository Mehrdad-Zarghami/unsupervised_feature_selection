import numpy as np
def weighted_distance(s1, s2, weight_vec, beta):
    """Calculate the weighted distance between two samples s1 and  s2 and based on the weights that each feature has

    Args:
        s1 (ndarray): _description_
        s2 (ndarray): _description_
        weight_vec (ndarray): a vector of size (n_features, ), each element is the weight of corresponding feature.
        beta (scaler): the power of weights vector

    Returns:
        scaler: the weighted distance between two samples
    """
    distance_vec = np.square(s1 - s2) # Element_wise --> for each feature
    # w**beta
    weights_beta_vector = np.power(weight_vec, beta)

    weighted_distance = np.dot(distance_vec, weights_beta_vector.T)
    return weighted_distance


def closest_center(sample, centers, weight_vector, beta):
    """it takes a sample and compare its distance to centers of clusters and return the cluster with closest center.

    Args:
        sample (ndarray): A vector that represent a data point
        centers (ndarray): A ndarray with the shape of (n_clusters, n_features) where each row represent a center of a cluster
        weight_vector (ndarray): a vector of wights for corresponding feature
        beta (scaler): a bridge that bring
    Returns:
        int: the number of cluster which is closest to the samples
    """
    d = [] # list of weighted distances
    for c in centers:
        w_d = weighted_distance(sample, c, weight_vector, beta)
        d.append(w_d)
    assigned_cluster = np.argmin(d)
    return assigned_cluster


def u_calculation(data, centers, weights, beta ):
   """ Calculate U based on Z and W and our dataset
   """
   n_spl = data.shape[0] # umber of samples
   n_clu = centers.shape[0] # number of clusters
   u_matrix = np.zeros((n_spl, n_clu))
   for i, x in enumerate(data):
      l = closest_center(x, centers, weights, beta)
      u_matrix[i,l] = 1
   return u_matrix


def clusters_vec(U):
    # a vector of size (n_samples, ) which each element shows the cluster that corresponding samples is assigned to
    n_samples = U.shape[0]
    c_vec = np.zeros(n_samples)
    for m, u in enumerate(U):
        c_vec[m] = np.argmax(u)
    c_vec = c_vec.astype(int)
    return c_vec

def cost_function(U, Z, W, X, beta ):
    """Calculate the cost function

    Args:
        U (ndarray):  U is an (M, k) matrix, ui,l is a binary variable, and ui,l = 1 indicates that record i is allocated to cluster l.
        Z (ndarray): is a set of k vectors representing the k-cluster centers of size (n_clusters, n_features)
        W (ndarray): W = [w1, w2, ..., wN ] is a set of weights of size (n_features, )
        X (ndarray): matrix of records (n_records, n_features)
        beta (int, optional): The power of elements of weights vector Defaults to 2.
    """
    P = 0 # initial value of cost

    cl_vec = clusters_vec(U)
    
    # Updating P
    for m, c in enumerate(cl_vec):
        w_d = weighted_distance(X[m], Z[c], W, beta)
        P += w_d
        P =  P.item() # to convert it to a single scaler
    return(P)



def clusters_dict(U):
    # Finding the index of samples in each cluster
    n_clusters = U.shape[1]
    cluster_dict = {}
    clu_vec = clusters_vec(U)

    for i in range(n_clusters):
        cluster_dict[i] = np.where(clu_vec == i)[0]
        
    return cluster_dict

def update_Z(U, Z, X):
    """Update Z i.e. the centers of clusters, by taking mean of teh samples in each cluster
    """
    cluster_dict = clusters_dict(U)

    new_Z = np.zeros_like(Z)
    for i,ind in cluster_dict.items():
        new_Z[i] = np.mean(X[ind], axis=0)
    
    return new_Z

def dj(X, U, Z):
    # Iteration over all features to calculate Dj for each feature
    
    cluster_dict = clusters_dict(U)
    n_features = X.shape[1]
    n_clusters = U.shape[1]
    
    D = []
    for j in range(n_features):
        d_j = 0
        for l in range(n_clusters):
            inx_in_cluster = cluster_dict[l]
            # Distance for feature "j" in cluster "l"
            d_j_l =np.sum(np.square(X[inx_in_cluster][:,j]-Z[l][j])) 
            ###################bug was here
            #d_j_l =np.sum(np.square(X[inx_in_cluster][j]-Z[l][j]))
            d_j += d_j_l

        D.append(d_j)
    return D


def weight_update(X, U, Z, weights, beta):
    # D calculation:
    D = dj(X, U, Z)
    # weights_update
    weights_upd = np.zeros_like(weights)

    # wherever D is zero, the corresponding weight is zero
    ind_D_zero = np.where(D == 0 )[0] # indexes of zero Dj
    weights_upd[ind_D_zero] = 0

    # D is not zero
    ind_D_not_zero = np.where(D)[0] ## indexes of non-zero Dj
    for j in ind_D_not_zero:
        
        Dj_Dt = 0
        for t in ind_D_not_zero:
            Dj_Dt += (D[j] / D[t]) ** (1 / ( beta - 1) )
        
        weights_upd[j] = 1 / Dj_Dt

    return weights_upd


#################################
# for subspace weighted k_means:


def sub_dj(X, U, Z):
    """ Iteration over all features to calculate D (dispersion) for each feature in each subspace or cluster

    Args:
        U (ndarray):  U is an (M, k) matrix, ui,l is a binary variable, and ui,l = 1 indicates that record i is allocated to cluster l.
        Z (ndarray): is a set of k vectors representing the k-cluster centers of size (n_clusters, n_features)
        X (ndarray): matrix of records (n_records, n_features)


    Returns:
        D(nd.array): a  matrix of size (n_clusters, n_features), where element [l,j] is the dispersion for cluster l and feature j.
    """

    
    cluster_dict = clusters_dict(U)
    n_features = X.shape[1]
    n_clusters = U.shape[1]
    
    D = np.empty((n_clusters,n_features)) # each row is for each cluster and each column is for each feature
    for l in n_clusters:
        inx_in_cluster = cluster_dict[l]
        for j in range(n_features):
            # Distance for feature "j" in cluster "l"
            D[l, j] =np.sum(np.square(X[inx_in_cluster][j]-Z[l][j]))

    return D

def sub_weight_update(X, U, Z, weights, beta):
    
    n_clusters = weights.shape[0]
    n_features = weights.shape[1]


    # D calculation:
    D = sub_dj(X, U, Z)

    # weights_update
    weights_upd = np.empty_like(weights) # a matrix of size (n_clusters, n_features)

    for l in n_clusters:
        for j in n_features:
            # calculation of sum of {( D[lj] / D[lt] ) ** (1 / ( beta - 1) )} for all t, 1 < t < n_features
            Dlj_Dlt = 0
            for t in n_features:
                Dlj_Dlt += (D[l,j] / D[l,t]) ** (1 / ( beta - 1) )
            
            weights_upd[l, j] = 1 / Dlj_Dlt

    return weights_upd


