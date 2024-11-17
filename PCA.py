import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(seed=42)

#Step 1: Generate dataset  in R2
mu=np.array([2,5])
cov=np.array([[1,0.9],
              [0.9,1]])
n=100
X=rng.multivariate_normal(mu,cov,n).T #dxn
d,n=X.shape
plt.subplot(1,2,1)
plt.scatter(X[0,:],X[1,:])
plt.axhline(color='black',linestyle='--') #draws horizontal line
plt.axvline(color='black',linestyle='--') #draws vertical line
plt.axis('equal')
plt.title('Original')

#Step 2: Center the dataset
mu=X.mean(axis=1) #mean along columns
X-=mu.reshape(2,1)
plt.subplot(1,2,2)
plt.scatter(X[0,:],X[1,:])
plt.axhline(color='black',linestyle='--') #draws horizontal line
plt.axvline(color='black',linestyle='--') #draws vertical line
plt.axis('equal')
plt.title('Centred')

plt.show()

#Step 3: Covariance Matrix
C=X@X.T/n

#Step 4: Principle Components Generation: Eigen vectors of C
eigval,eigvec=np.linalg.eigh(C)
var=np.flip(eigval) #Converts ascending to descending and vice versa
pcs=np.flip(eigvec, axis=1) #flips the columns
#eigenvalues are the variances(var)
#eigenvectors are the principle components(pcs)

#Step 5: Vizualise the PCs
#PCA 1: w_1=(w_11,w_12)
#TO draw a line passing through w_1 the slope m_1 of the line should be w_12/w_11
m_1=pcs[:,0][1]/pcs[:,0][0] #PC 1
m_2=pcs[:,1][1]/pcs[:,1][0] #PC 2
x=np.linspace(-2,2)
plt.scatter(X[0,:],X[1,:],alpha=0.5) #Alpha gives transparency
plt.axhline(color='black',linestyle='--') #draws horizontal line
plt.axvline(color='black',linestyle='--') #draws vertical line
plt.axis('equal')
plt.plot(x,m_1*x,color='green',label='PC 1')
plt.plot(x,m_2*x,color='red',label='PC 2')
plt.legend()
plt.show()

#Function for PCA
def PCA(X):
    d,n=X.shape
    X-=X.mean(axis=1).reshape(d,1)
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
    
#Implementation using functions defined
mu=np.array([2,5])
cov=np.array([[1,0.5],
              [0.5,1]])
n=100
X=rng.multivariate_normal(mu,cov,n).T
var,pcs=PCA(X)
plot(X,pcs)