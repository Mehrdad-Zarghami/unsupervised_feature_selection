#Function to compute correlation between two variables
#method:
# 1 = Feature Similarity: Correlation Coeff 
# 2 = Feature Similarity: Linear Regression error
# 3 = Feature Similarity: Maximal Information Compression Index
import numpy as np
def f2f(x1,x2, method):
    no_x1=np.shape(x1)[0]
    no_x2=np.shape(x2)[0]
    dist=0.0
    if method == 1:
        pass
    if method == 2:
        pass
    if method == 3:
        sxy=0.0; sx=0.0; sy=0.0; mnx1=0.0; mnx2=0.0;
        for i in range(no_x1):
            sxy=sxy+x1[i]*x2[i]
            sx=sx+x1[i]**2
            sy=sy+x2[i]**2
            mnx1=mnx1+x1[i]
            mnx2=mnx2+x2[i]
        mnx1=mnx1/no_x1;
        mnx2=mnx2/no_x2;

        sxy=(sxy/no_x1)- mnx1*mnx2;
        sx=(sx/no_x1)-mnx1**2
        sy=(sy/no_x1)-mnx2**2

        if (sx-sy) ==0:
            theta=0.5*np.pi/2
        else:
            theta=0.5*np.arctan(2*sxy/(sx-sy))

        a = -1 / np.tan(theta)
        b = np.mean(x1)*(-a) - np.mean(x2)

        dist = 0.0
        for i in range(no_x1):
            dist=dist+abs((x2[i]-a*x1[i]-b) / np.sqrt(a**2 + b**2))
        dist = dist/no_x1
        return dist 