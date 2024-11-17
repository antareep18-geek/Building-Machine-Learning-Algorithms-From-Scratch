import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=[6,6]

#### DATA: DO NOT EDIT THIS CELL ####
X = np.array([[1, -3], [1, 0], [4, 1], [3, 7], [0, -2],
             [-1, -6], [2, 5], [1, 2], [0, -1], [-1, -4],
             [0, 7], [1, 5], [-4, 4], [2, 9], [-2, 2],
             [-2, 0], [-3, -2], [-2, -4], [3, 10], [-3, -8]]).T
y = np.array([1, 1, 1, 1, 1,
             1, 1, 1, 1, 1,
             -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1])

#Visualising the data
d,n=X.shape
colors=np.array(['','green','red'])
plt.scatter(X[0,:],X[1,:],c=colors[y])
plt.axhline(color='black',linestyle='--')
plt.axvline(color='black',linestyle='--')
plt.show()

#Linear Separability : Perceptron : If converges then L.separable else not L.separable
w=np.zeros(d)
cycle=0
while cycle<n:
    for i in range(n):
        #Prediction for the ith datapoint
        if w@X[:,i]>=0:
            y_hat=1
        else:
            y_hat=-1
        if y_hat!=y[i]:
            cycle=0
            w+=X[:,i]*y[i]
        else: 
            cycle+=1
print(w)

colors=np.array(['','green','red'])
plt.scatter(X[0,:],X[1,:],c=colors[y])
plt.axhline(color='black',linestyle='--')
plt.axvline(color='black',linestyle='--')
plt.arrow(0,0,w[0],w[1],head_width=0.5, head_length=0.5)
x=np.linspace(-5,5)
plt.plot(x,-w[0]*x/w[1],linestyle='--',color='black')
plt.axis('equal')
plt.show()

#Hard-Margin SVM on Dual
Y=np.diag(y)
Q=Y.T@X.T@X@Y
def f(alpha):
    return 0.5*alpha.T@Q@alpha-alpha.sum()
from scipy.optimize import minimize
from scipy.optimize import Bounds
res = minimize(f,x0=np.zeros(n),bounds=Bounds(0,np.inf))
alpha_star=res.x

X_sup = X[:, alpha_star > 0]
y_sup = y[alpha_star > 0]
y_sup_color = np.where(y_sup == 1, 'green', 'red')

w_star=X@Y@alpha_star
colors=np.array(['','green','red'])
plt.scatter(X[0,:],X[1,:],c=colors[y])
plt.axhline(color='black',linestyle='--')
plt.axvline(color='black',linestyle='--')
plt.arrow(0,0,w_star[0],w_star[1],head_width=0.5, head_length=0.5)
x=np.linspace(-5,5)
plt.plot(x,-w_star[0]*x/w_star[1],linestyle='--',color='black')
plt.plot(x,1/w_star[1]-w_star[0]*x/w_star[1],linestyle='--',color='black')
plt.plot(x,-1/w_star[1]-w_star[0]*x/w_star[1],linestyle='--',color='black')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.show()

#Soft-Margin SVM on dual
def train_SVM(X,y,C):
    d,n=X.shape
    Y=np.diag(y)
    Q=Y.T@X.T@X@Y
    f=lambda alpha: 0.5*alpha.T@Q@alpha-alpha.sum()
    bounds = Bounds(np.zeros(n), C * np.ones(n))
    res = minimize(f, np.zeros(n), bounds=bounds)
    alpha_star=res.x
    w_star=X@Y@alpha_star
    return w_star,alpha_star


def plot(X,y,w_star):
    colors=np.array(['','green','red'])
    plt.scatter(X[0,:],X[1,:],c=colors[y])
    plt.axhline(color='black',linestyle='--')
    plt.axvline(color='black',linestyle='--')
    plt.arrow(0,0,w_star[0],w_star[1],head_width=0.5, head_length=0.5)
    x=np.linspace(-5,5)
    plt.plot(x,-w_star[0]*x/w_star[1],linestyle='--',color='black')
    plt.plot(x,1/w_star[1]-w_star[0]*x/w_star[1],linestyle='--',color='black')
    plt.plot(x,-1/w_star[1]-w_star[0]*x/w_star[1],linestyle='--',color='black')
    plt.xlim(-10,10)
    plt.ylim(-10,10)

for i,C in enumerate([0.01,0.1,1,10]):
    w_star,alpha_star=train_SVM(X,y,C)
    plt.subplot(2,2,i+1)
    plot(X,y,w_star)
    plt.title(f'C={C}')
plt.show()
