import numpy as np
import matplotlib.pyplot as plt
import cv2
rng=np.random.default_rng(seed=138)
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

img=cv2.imread('cube.jpg')
img=cv2.resize(img,(100,100))
plt.imshow(img)
plt.show()
X=img.reshape(10_000,3).T
X=X.astype(float)
z=k_means(X,k=4)
d,n=X.shape
#Each pixel can be identified with one of the four colors
color_mapping={0:[255,0,0],1:[0,255,0],2:[0,0,255],3:[255,255,255]}
for i in range(X.shape[1]):
    X[:,i]=color_mapping[z[i]]
img_updated=X.T.reshape(100,100,3)
plt.imshow(img_updated.astype(int))
plt.show()