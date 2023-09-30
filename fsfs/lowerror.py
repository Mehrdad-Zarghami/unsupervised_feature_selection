# Function to find the lowest error

import numpy as np

def lowerror(dm, k):
    no_feature = dm.shape[0]
    HIGH = 9999
    kd = np.zeros(no_feature)

    for i in range(no_feature):
        if dm[i, i] == 1:
            kd[i] = HIGH
        else:
            if i == 0:
                dd = np.concatenate(([HIGH], dm[0, 1:]))
            elif i == no_feature - 1:
                dd = np.concatenate((dm[0:no_feature - 1, no_feature - 1], [HIGH]))
            else:
                dd = np.concatenate((dm[0:i, i], [HIGH], dm[i, i + 1:no_feature]))
            
            for l in range(no_feature):
                if dm[l, l] == 1:
                    dd[l] = HIGH
            
            dd = np.sort(dd)
            kd[i] = dd[k]

    erro, indx = np.min(kd), np.argmin(kd)
    return erro, indx

            