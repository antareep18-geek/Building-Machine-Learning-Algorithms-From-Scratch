import numpy as np
import matplotlib.pyplot as plt

#Generate dataset
rng=np.random.default_rng(seed=138)
mus=np.array([[-3,3],
             [3,3],
             [3,-3]])
cov=np.eye(2)
n=60
X=np.concatenate([rng.multivariate_normal(mus[j],cov,n//3) for j in range(3)],axis=0).T

#Visualise the dataset
plt.scatter(X[0,:],X[1,:])
plt.show()

#mus[:,0]=mean of clus 1
#mus[:,1]=mean of clus 2
#mus[:,2]=mean of clus 3

#K-means function
def k_means(X,k):
    #i=data pts
    #j=clusters
    #Step 1: Initialisation
    d,n=X.shape
    ind=rng.choice(np.arange(n),k,replace=False) #choosing 3 different data pts from datasets and storing their indices
    mus=X[:,ind] #the 3 random points from prev line are assigned as means
    z=np.ones(n)
    z_prev=np.zeros(n)
    while not np.array_equal(z,z_prev):
        z_prev=z.copy() #deep copy: z will not change on changing z_prev  
        #Step 2: Cluster Assignment
        for i in range(n):
            dist=np.linalg.norm(mus-X[:,i].reshape(d,1),axis=0) #column-wise norm to get the eucledian distances
            z[i]=np.argmin(dist)
        #Step 3: Re-assigning Cluster Centres
        for j in range(k):
            if np.any(z==j):#check for whether the slice is empty or not. Returns true if at least one is true
                mus[:,j]=X[:,z==j].mean(axis=1)
    return z

#plotting the clusters with different colors
z=k_means(X,3)
colors=np.array(['red','blue','green'])
plt.scatter(X[0,:],X[1,:],c=colors[z.astype(int)])
plt.show()