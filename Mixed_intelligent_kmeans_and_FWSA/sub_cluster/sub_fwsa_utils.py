# the utils file for fwsa --> modification of utils for wkmeans
import numpy as np

def center_of_data(X):
    '''
    calculate the mean of each feature for dataset and gives out a sigle point which is the center of the dataset X

    Parameters
    ----------
    x : TYPE
        The dataset (m,n) , m: the number of samples in dataset; n the number of features for each sample

    Returns
    -------
    a single point "g"
    which is the center(mean) of the dastaset
    '''
    g = np.mean(X, axis=0)
    return g


def weighted_distance(s1, s2, weight_vec):
    """Calculate the weighted distance between two samples s1 and  s2 and based on the weights that each feature has

    Args:
        s1 (ndarray): _description_
        s2 (ndarray): _description_
        weight_vec (ndarray): a vector of size (n_features, ), each element is the weight of corresponding feature.

    Returns:
        scaler: the weighted distance between two samples
    """
    distance_vec = np.square(s1 - s2) # Element_wise --> for each feature
    weighted_distance = np.dot(distance_vec, weight_vec.T)
    return weighted_distance



def closest_center(sample, centers, weight_vector):
    """it takes a sample and compare its distance to centers of clusters and return the cluster with closest center.

    Args:
        sample (ndarray): A vector that represent a data point
        centers (ndarray): A ndarray with the shape of (n_clusters, n_features) where each row represent a center of a cluster
        weight_vector (ndarray): a vector of wights for corresponding feature
    Returns:
        int: the number of cluster which is closest to the samples
    """
    d = [] # list of weighted distances
    for c in centers:
        w_d = weighted_distance(sample, c, weight_vector)
        d.append(w_d)
    assigned_cluster = np.argmin(d)
    return assigned_cluster


def u_calculation(data, centers, weights):
   """ Calculate U based on Z and W and our dataset
   """
   n_spl = data.shape[0] # umber of samples
   n_clu = centers.shape[0] # number of clusters
   u_matrix = np.zeros((n_spl, n_clu))
   for i, x in enumerate(data):
      l = closest_center(x, centers, weights)
      u_matrix[i,l] = 1
   return u_matrix

def clusters_vec(U):
    """
    a vector of size (n_samples, ) which each element shows the cluster that corresponding samples is assigned to
    """

    n_samples = U.shape[0]
    c_vec = np.zeros(n_samples)
    for m, u in enumerate(U):
        c_vec[m] = np.argmax(u)
    c_vec = c_vec.astype(int)
    return c_vec




def clusters_dict(U):
    # Finding the index of samples in each cluster
    n_clusters = U.shape[1]
    cluster_dict = {}
    clu_vec = clusters_vec(U)

    for i in range(n_clusters):
        cluster_dict[i] = np.where(clu_vec == i)[0]
        
    return cluster_dict

def update_Z(U, Z, X, one_center_fixed = False, center_of_original_dataset = None):
    """Update Z i.e. the centers of clusters, by taking mean of teh samples in each cluster

        When to use it for intelligent_FWSA, we need to keep one center fixed so "one_center_fixed" should be "True" and 
        also we should proved the "center_of_original_dataset" 
    
    """
    cluster_dict = clusters_dict(U)
    new_Z = np.zeros_like(Z)
    if one_center_fixed == False:
        for i,ind in cluster_dict.items():
            new_Z[i] = np.mean(X[ind], axis=0)
    else:
        new_Z[0] = center_of_original_dataset # The first cluster center is the center of Original dataset
        for i,ind in cluster_dict.items():
            if i == 0:
                continue # skip calculating the mean of points for first cluster 
            new_Z[i] = np.mean(X[ind], axis=0)
        
    return new_Z


#################################
#for fwsa
def a_inner_cluster_seperation(U, Z, X):
    '''
    Parameters
    ----------
    U : ndarray(M,k) --> (no.samples, no.clusters)
        (M, k) matrix, u(i,l) is a binary variable where 1 indicates that record i is allocated to cluster l.
    Z : ndarray(k,N) --> (no.clusteres, no.features)
        is a set of k vectors representing the k-cluster centers of size
    X : ndarray(no.recordes) --> (n_records, n_features)
        matrix of records (n_records, n_features)

    Returns
    -------
    a : ndarray(1,N) --> (1, no.features)
        the sum of separations within clusters for each feature.

    '''
    UZ = np.matmul(U ,Z) # a matrix (no.samples , no.features) where row i is the center of cluster that sample i belonges to.
    
    inner_dissimilarity = np.square(X - UZ)
    a = np.sum(inner_dissimilarity,axis=0)
    return a

def b_between_clusters_sepperation(U, Z, X):
    '''
    the sum of separations between clusters

    Parameters
    ----------
    U : ndarray(M,k) --> (no.samples, no.clusters)
        (M, k) matrix, u(i,l) is a binary variable where 1 indicates that record i is allocated to cluster l.
    Z : ndarray(k,N) --> (no.clusteres, no.features)
        is a set of k vectors representing the k-cluster centers of size
    X : ndarray(no.recordes) --> (n_records, n_features)
        matrix of records (n_records, n_features)

    Returns
    -------
    b : ndarray(1,N) --> (1, no.features)
        the sum of separations between clusters 

    '''
    clusters_cardinality = np.sum(U, axis=0) # a vector(1, no.clusters) where each element is the cardinality of cluster k
    g = center_of_data(X) 
    clusters_dissimilarity = np.square(Z - g)
    b = np.dot(clusters_cardinality, clusters_dissimilarity)
    return b
    
    
def fwsa_cost_function(U, Z, W, X):
    '''
    Parameters
    ----------
   U (ndarray):  (M, k) matrix, u(i,l) is a binary variable where 1 indicates that record i is allocated to cluster l.
   Z (ndarray): is a set of k vectors representing the k-cluster centers of size (n_clusters, n_features)
   W (ndarray): W = [w1, w2, ..., wN ] is a set of weights of size (n_features, )
   X (ndarray): matrix of records (n_records, n_features)

    Returns
    -------
    cost : scaler(float)
        sum(W_n * b_n) / sum(W_n * a_n)

    '''
    a = a_inner_cluster_seperation(U, Z, X) # --> the sum of separations within clusters for each feature
    b = b_between_clusters_sepperation(U, Z, X) # -->  (1, no.features) the sum of separations between clusters for each feature 
    
    cost = np.dot(W, b) / np.dot(W, a)
    return cost

def fwsa_weight_update(X, U, Z, weights):
    '''
    (1/2) * ( W(t) + delta_W(t) )
    Where: delta_W_n(t) = (b_n / a_n) / sum_over_n (b_n / a_n)
    
    '''
    a = a_inner_cluster_seperation(U, Z, X)
    b = b_between_clusters_sepperation(U, Z, X)
    sum_b_over_a = np.sum(b / a)
    dw  = (b/a) / sum_b_over_a
    updated_W = 0.5 * (weights + dw)

    return updated_W


#################################
# For sub_fwsa


def sub_closest_center(sample, centers, weight_matrix):
    """it takes a sample and compare its distance to centers of clusters and return the cluster with closest center.

    Args:
        sample (ndarray): A vector that represent a data point
        centers (ndarray) (n_clusters, n_features): A ndarray with the shape of (n_clusters, n_features) where each row represent a center of a cluster
        weight_matrix (ndarray) (n_clusters, n_features): a vector of wights for corresponding feature in corresponding cluster
    Returns:
        int: the number of cluster which is closest to the samples
    """
    d = [] # list of weighted distances
    for c, w  in zip(centers,weight_matrix): #--> corresponding set of weights for center of each cluster
        w_d = weighted_distance(sample, c, w)
        d.append(w_d)
    assigned_cluster = np.argmin(d)
    return assigned_cluster


def sub_u_calculation(data, centers, weight_matrix):
   """ Calculate U based on Z and W and our dataset
   """
   n_spl = data.shape[0] # umber of samples
   n_clu = centers.shape[0] # number of clusters
   u_matrix = np.zeros((n_spl, n_clu))
   for i, x in enumerate(data):
      l = sub_closest_center(x, centers, weight_matrix)
      u_matrix[i,l] = 1
   return u_matrix


def sub_a_inner_cluster_seperation(U, Z, X):
    '''
    Parameters
    ----------
    U : ndarray(M,k) --> (no.samples, no.clusters)
        (M, k) matrix, u(i,l) is a binary variable where 1 indicates that record i is allocated to cluster l.
    Z : ndarray(k,N) --> (no.clusters, no.features)
        is a set of k vectors representing the k-cluster centers of size
    X : ndarray(no.records) --> (n_records, n_features)
        matrix of records (n_records, n_features)

    Returns
    -------
    a : ndarray (no_clusters, no.features)
        each row is for each cluster

    '''

    cluster_dict = clusters_dict(U)
    a = np.zeros_like(Z)

    for i,ind in cluster_dict.items():
        a[i] = np.sum(np.square(X[ind] - Z[i]), axis=0)

    return a

def sub_b_between_clusters_sepperation(U, Z, X):
    '''
    the sum of separations between clusters

    Parameters
    ----------
    U : ndarray(M,k) --> (no.samples, no.clusters)
        (M, k) matrix, u(i,l) is a binary variable where 1 indicates that record i is allocated to cluster l.
    Z : ndarray(k,N) --> (no.clusters, no.features)
        is a set of k vectors representing the k-cluster centers of size
    X : ndarray(no.records) --> (n_records, n_features)
        matrix of records (n_records, n_features)

    Returns
    -------
    b : ndarray(no_clusters, no.features)
        each row is for each cluster: a notion of separations between clusters 

    '''
    clusters_cardinality = np.sum(U, axis=0) # a vector(1, no.clusters) where each element is the cardinality of cluster k
    n_clu = Z.shape[0] # number of clusters
    g = center_of_data(X)

    b = np.zeros_like(Z)

    for i in range(n_clu):
        b[i] = clusters_cardinality[i] * np.square(Z[i] - g)

    return b



def sub_fwsa_cost_function(U, Z, W_matrix, X):
    '''
    Parameters
    ----------
   U (ndarray):  (M, k) matrix, u(i,l) is a binary variable where 1 indicates that record i is allocated to cluster l.
   Z (ndarray): is a set of k vectors representing the k-cluster centers of size (n_clusters, n_features)
   W_matrix (ndarray): matrix of weights of size (n_clusters, n_features) where for cluster (row) k, we have [w_k1, w_k2, ..., w_kN ] where N is the number of features
   X (ndarray): matrix of records (n_records, n_features)

    Returns
    -------
    cost : scaler(float)
        sum_over_k( sum_over_v ( w_kv * b_kv)) / sum_over_k( sum_over_v ( w_kv * a_kv))

    '''
    a = sub_a_inner_cluster_seperation(U, Z, X) # --> the sum of separations within clusters for each feature
    b = sub_b_between_clusters_sepperation(U, Z, X) # -->  (1, no.features) the sum of separations between clusters for each feature 
    
    cost_num = 0 #--> numerator of cost function: sum_over_k ( sum_over_v ( w_kv * b_kv))
    cost_denum = 0 #--> denominator of cost function: sum_over_k ( sum_over_v ( w_kv * a_kv))
    n_clu = Z.shape[0] # number of clusters
    for i in range(n_clu):
        cost_num += np.dot(W_matrix[i], b[i])
        cost_denum += np.dot(W_matrix[i], a[i])
    cost = cost_num / cost_denum
    return cost

def sub_fwsa_weight_update(X, U, Z, weights_matrix):
    '''
    (1/2) * ( W(t) + delta_W(t) )
    Where: delta_W_n(t) = (b_n / a_n) / sum_over_n (b_n / a_n)
    
    '''
    a = sub_a_inner_cluster_seperation(U, Z, X)
    b = sub_b_between_clusters_sepperation(U, Z, X)
    sum_b_over_a = np.sum(b / a, axis=1).reshape(-1,1)
    dw  = (b/a) / sum_b_over_a
    updated_W = 0.5 * (weights_matrix + dw)

    return updated_W


    
    
    
    



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


