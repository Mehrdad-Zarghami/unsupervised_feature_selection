## fsfs
def fsfs(data,k, method=3, original_size = None):
    """
    Reduces a feature set using the method described in:P. Mitra, C. A. Murthy and S. K. Pal,
    Unsupervised Feature Selection using Feature Similarity, IEEE Transactions on Pattern Analysis and Machine Intelligence,
    Vol .24, No. 4, pp 301-312, April 2002. 


    # Args

    #     data: Data matrix containing the original feature set. Each column represents a feature. In this Matrix, all columns are input features (There is no target column).
    #     k: Scale parameter which decides the size of the reduced feature set. Approximately, k = original_size - size of the reduced feature set.
    #     method: 1: Lasso Regression, 2: Ridge Regression, 3: Elastic Net Regression.
    #     original_size: Number of features in the original data set. It seems it is a redundent input argument, becuse we can calculate the number of features just by counting the number of columns in data matrix.


    # Returns:
    
    #     redu : Reduced features. A vector containing the feature numbers of the original feature set which are retained in the reduced set.
    #     fwt : feature weights of the features in redu. 

    """
    
    if original_size == None:
        original_size = data.shape[1]
    no_feature=original_size;
    no_data=np.shape(data)[0]

    kk=[]
    while k > (no_feature-1):

        print('Give a smaller value of k\n')
        k=input('k=  ')


    # Form the inter-feature distance matrix (Upper Triangular)
    """
    for i=1:no_feature,
       fprintf(1,'Similarity Computed for Feature No. %d\n',i);
       for j=1:no_feature,
          x1=data(:,i);x2=data(:,j);
          if i < j
             dm(i,j)=f2f(x1,x2,method);
          else
             dm(i,j)=0.0;
          end
       end
    end
    """
    print('Computing Feature Similarities..\n')
    dm  = np.zeros((no_feature,no_feature))

    for i in range (no_feature):
        print(f'Similarity Computed for Feature No.{i}')
        # dm[i,i] = 1 # --> not implemented in the original code, nut we ned it for calculating the trace  of dm, which is iqual to the number of features in dm
        for j in range (no_feature):
            x1=data[:,i]
            x2=data[:,j]


            if i < j:
                dm[i,j] = f2f(x1,x2, method)
            elif i > j:
                dm[i,j] = 0
    print('*********************************************')
    print(f"dm matrix is: \n",dm)
    print('*********************************************')
    drift = 1.0

    # Form a vector containing the distance of k-NN for each feature.
    """  

    """
    print('Vectores containing the distance of k-NN for each feature')
    kd = np.empty(no_feature)
    for i in range (no_feature):
        if i==0:
            dd=dm[0,1:]
        elif i == no_feature - 1:
            dd = dm[:no_feature - 1, no_feature - 1].T
        else:
            dd = np.concatenate((dm[i, i + 1:], dm[:i, i].T))
        dd = sorted(dd)
        kd[i]  = dd[k-1] # --> Becuse of 0-indexed vectors in python, 
        print(f'dd for feature NO. {i}:')
        print(dd)
    print('**************************') 
    print("kd vector is: \n",kd)  

    kd0 = kd #--> Store the original r_k

    # Condense the feature set
    print('**************************') 
    print('Clustring the features...\n')
    rfs=[]; rfd=[]; ee=[];dmt=dm; lower=9999; iter=0; prev_lower=9999;
    tagm=np.ones(no_feature)
    while (no_feature - np.trace(dm)) > 0:
        iter = iter + 1
        if k > (no_feature - np.trace(dm) - 1):
            k = (no_feature - np.trace(dm) - 1)
        if k<= 1: #--> it was ( <= 0 )
            break
        prev_lower = lower
        lower,fetr = lowerror(dm,k)

        #Adjust k
        while lower > drift*prev_lower:
            k = k-1
            if k == 0:
                break
            lower, fetr = lowerror(dm,k)


        if k <= 0:
            break
        dm = updatedm(dm,fetr,k) # --> need to be implemented 


        # kk=[kk;k];
        # ee=[ee;lower];
        kk.append(k)
        ee.append(lower)

        tagm[fetr]=0
        # for i=1:no_feature,
        #     for j=1:no_feature,
        #         if dm(i,i)==1
        #             tagm(i)=0;
        #         end
        #     end
        # end
        for i in range(no_feature):
            for j in range(no_feature):
                if dm[i,i] == 1: # -------------> shouldn't it be dm[i,j]??????? 
                    tagm[i] = 0
    
    # for i=1:no_feature,
    #     if dm(i,i)==0
    #         rfs=[rfs;i];
    #         rfd=[rfd;kd0(i)];
    #     end
    # end
    for i in range(no_feature):
        if dm[i,i] == 0: # --> we already set dm[i,i] = 1 for the trace purpose
            rfs.append(i)  
            rfd.append(kd0[i])

    print('Features Clustered.\n')
    redu=rfs
    fwt=rfd

    return redu, fwt

if __name__ == "__main__":
    from f2f import f2f
    from lowerror import lowerror
    from updatedm import updatedm 
    import numpy as np
    # Set the maximum number of columns to display
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    def normalizer(data):
        return (data - np.mean(data,axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    
    # Importing a dataset with high number of features
    from sklearn.datasets import fetch_california_housing

    # Load the California Housing dataset
    california_housing = fetch_california_housing()

    # Access the features and target
    X = california_housing.data  # Features
    y = california_housing.target  # Target variable (median house value)
    X = normalizer(X)

    tmp = fsfs(X, 3, method=3)
    print(tmp)
