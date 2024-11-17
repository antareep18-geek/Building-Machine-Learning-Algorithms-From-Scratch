import numpy as np
import matplotlib.pyplot as plt

#Data generation
rng=np.random.default_rng(seed=42)
n=20
x=np.linspace(-1,1,n)
p=8
X=np.array([x**i for i in range (p+1)])
y_true=np.sin(2*np.pi*x)
y=y_true+rng.normal(0,0.2,n)
print(X.shape)
print(y.shape) 

#Plotting the data
plt.scatter(x,y)

#Plotting the true function
x_vis=np.linspace(-1,1,100)
y_true_vis=np.sin(2*np.pi*x_vis)
plt.plot(x_vis,y_true_vis,linestyle='--',color='black',label='True')

#Regression
d,n=X.shape
lambd=0.00001 #regularization
w=np.linalg.pinv(X@X.T+lambd*np.eye(d))@X@y
print(w)

#Plotting the prediction
y_pred_vis=np.array([w[i]*(x_vis**i) for i in range(p+1)]).sum(axis=0)
plt.plot(x_vis,y_pred_vis,color='red',label='Prediction')
plt.legend()
plt.show()

#Loss function
def loss(X,y,w):
    y_pred=X.T@w
    error=(y-y_pred)
    mean_sq_err=(error**2).mean()
    return mean_sq_err

print('error=',loss(X,y,w))

#Finding eta
singualar_values=np.linalg.svd(X)[1]
sigma_1=singualar_values[0]
eta_max=2/sigma_1**2
print('eta max',eta_max)
#eta should be less than eta_max
#Here we get eta=0.07400384596691366

#Optimisation : Gradient Descent
w=np.zeros(d)
eta=0.074
loss_history=[0,np.inf] #np.inf is basically infinity
tolerance=1e-10
iter=0
while abs(loss_history[-1]-loss_history[-2])>tolerance:
    loss_history.append(loss(X,y,w))
    grad=X@X.T@w-X@y
    w-=eta*(grad/np.linalg.norm(grad))
    iter+=1
print(f'GD terminates after {iter} iterations')
print('final loss=',loss_history[-1])
print('final weight vector=',w)

plt.plot(range(iter),loss_history[2:])
plt.show()