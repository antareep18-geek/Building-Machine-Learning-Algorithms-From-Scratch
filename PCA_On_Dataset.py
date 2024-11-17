import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(seed=42)

#Function for PCA
def PCA(X):
    d,n=X.shape
    X=X-X.mean(axis=1).reshape(d,1)
    C=X@X.T/n
    eigval,eigvec=np.linalg.eigh(C)
    var=np.flip(eigval)
    pcs=np.flip(eigvec,axis=1)
    return var,pcs

#Function for plotting
def plot(X,pcs):
    d,n=X.shape
    plt.scatter(X[0,:],X[1,:],alpha=0.5) #Alpha gives transparency
    plt.axhline(color='black',linestyle='--') #draws horizontal line
    plt.axvline(color='black',linestyle='--') #draws vertical line
    m_1=pcs[:,0][1]/pcs[:,0][0] #PC 1
    m_2=pcs[:,1][1]/pcs[:,1][0] #PC 2
    x=np.linspace(np.min(X[0,:]),np.max(X[0,:]))
    plt.axis('equal')
    plt.plot(x,m_1*x,color='green',label='PC 1')
    plt.plot(x,m_2*x,color='red',label='PC 2')
    plt.legend()
    plt.show()
    
#Applying PCA on keras dataset    
from keras.datasets import mnist
train, test=mnist.load_data()
X_train,y_train=train
n=200
y_1=3 #Extracting data for label 3
y_2=8 #Extracting data for label 8
X=np.concatenate((X_train[y_train==y_1][:n//2],
                X_train[y_train==y_2][:n//2]),
                axis=0)
#Created a dataset by stacking of 100 3s and 100 8s
#first 100 are 3s
#next 100 are 8s
y=np.concatenate((np.zeros(n//2),np.ones(n//2)))
X=X.reshape(n,-1).T
var,pcs=PCA(X)
for k in range(784):
    if var[:k].sum()/var.sum()>=0.95:
        break
print(k)
#Gives the no. of PCss req to capture 95% of the variances

#Projection on PCs
W=pcs[:,:2] 
X_prime=W.T@X
a,b=X_prime.shape
colors=np.array(['red','green'])
plt.scatter(X_prime[0,:],X_prime[1,:],alpha=0.5,c=colors[y.astype(int)]) #astype does type casting
plt.xlim(a,b) #Zooming in
plt.show()
for i in range(n):
    if((-180<X_prime[0,i]<-140) and (-300<X_prime[1,i]<0)):# got these limits from visualising the scatter plot
        print(i) 
plt.subplot(1,2,1)  
plt.imshow(X[:,27].reshape(28,28),cmap='gray')
plt.subplot(1,2,2)  
plt.imshow(X[:,110].reshape(28,28),cmap='gray')
plt.show()