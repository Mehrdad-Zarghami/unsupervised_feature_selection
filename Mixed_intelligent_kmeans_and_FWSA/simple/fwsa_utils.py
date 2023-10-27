# the utils file for fwsa --> modification of utils for wkmeans
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

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

def update_Z(U, Z, X, one_center_fixed = False,center_of_original_dataset = None):
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
    UZ = np.matmul(U ,Z) # a matrix (no.samples , no.features) where row i is the center of cluster that sample i belongs to.
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


def dataset_selection():
    script_location = Path(__file__).resolve()
    # Access the folder containing the script
    
    script_folder = script_location.parent



    while(True):
            
        selection = input('Which data? Only enter the corresponding number:\n 1-Australian_credit \n 2-Balance_scaler\n 3-cancer\n 4-car \n 5-ecoli \n 6-glass\n 7-heart \n 8-ionosphere\n 9-iris\n 10-soya \n 11-teaching_assistant \n 12-tic+tac+toe \n 13-wine\n')
        if selection == '1':
            number_of_real_classes = 2
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_AustraCC.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_AustraCC.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)

            break
        if selection == '2':
            number_of_real_classes = 2
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_Balance.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_Balance.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)

            break

        if selection == '3':
            number_of_real_classes = 2
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_BreastCancer.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_BreastCancer.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)

            break

        elif selection == '4':
            number_of_real_classes = 4

            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_CarEvaluation.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_CarEvaluation.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)
            break

        elif selection == '5':
            number_of_real_classes = 8
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_Ecoli.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_Ecoli.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)
            break
        
        elif selection == '6':
            number_of_real_classes = 6
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_Glass.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_Glass.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)
            break

        elif selection == '7':
            number_of_real_classes = 2
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_Heart.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_Heart.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)
            break
        
        elif selection == '8':
            number_of_real_classes = 2
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_Ionosphere.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_Ionosphere.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)
            break

        elif selection == '9':
            number_of_real_classes = 3
            # Load the Iris dataset
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_Iris.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_Iris.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)

            break

        elif selection == '10':
            number_of_real_classes = 4
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_Soya.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_Soya.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)
            break

        elif selection == '11':
            number_of_real_classes = 3
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_TeachingAssistant.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_TeachingAssistant.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)
            break

        elif selection == '12':
            number_of_real_classes = 2
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_TicTacToe.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_TicTacToe.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)
            break
        elif selection == '13':
            number_of_real_classes = 3
            # Load the CSV file using Pandas
            X_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'X_Wine.csv'
            y_selected_dataset_directory = script_folder / 'data'/ 'Renato_data'/ 'y_Wine.csv'

            df_X = pd.read_csv(X_selected_dataset_directory,header=None)
            x = df_X.astype(float).values

            df_y = pd.read_csv(y_selected_dataset_directory,header=None)
            y = df_y.astype(float).values.reshape(-1,)
            break
        else:
            print("Invalid option")
    return x, y, number_of_real_classes



