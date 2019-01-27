### PLASSIER Vincent : Master M2 MVA 2017/2018 - Graphical models - HWK 3

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2
from scipy.linalg import sqrtm

plt.ion()
plt.show()


# on charge les données d'entraînement
u_train = np.loadtxt('D:\Cours M2_MvA\Probabilistic Graphical Models\DM3\classification_data_HWK3/EMGaussian.data')

# on charge les données de test
u_test = np.loadtxt('D:\Cours M2_MvA\Probabilistic Graphical Models\DM3\classification_data_HWK3/EMGaussian.test')

u = u_train # on peut choisir u_test
u1, u2 = u.T


K = 4 # nombre d'états possibles pour q
T, d = np.shape(u)


## Q1 : Implementation - HMM
# u : observations(K+1)
# mu : contient les centres des clusters
# cov : contient les matrices de covariances

func = lambda u, mu, cov : np.array([stats.multivariate_normal.logpdf(u, mu[i, :], cov[i]) for i in range(K)]) 

    
def forward(T,K,u,pi,mu,cov,A):
    alpha = np.zeros((T,K))
    alpha[0,:] = np.log(pi)+func(u[0],mu,cov)
    a = np.max(alpha[0,:])
    for t in range(T-1):
        alpha_bis = np.tensordot(alpha[t,:]-a,np.ones(K),0)
        alpha[t+1,:] = np.log(np.sum(np.exp(alpha_bis+np.log(A)), axis=0))+func(u[t+1],mu,cov)+a
        a = np.max(alpha[t+1,:])
    return alpha

def backward(T,K,u,pi,mu,cov,A):
    beta = np.zeros((T,K))
    b = 0
    for t in range(T-1, 0, -1):
        matrice = np.tensordot(np.ones(K),beta[t,:]-b+func(u[t],mu,cov),0)
        beta[t-1,:] = np.log(np.sum(np.exp(matrice+np.log(A)), axis=1))+b
        b = np.max(beta[t-1,:])
    return beta


## 3 - Implementation de l'algorithme EM


  ## a) - EM_GMM + Initialisation des paramètres mu, tau, pi, cov, A
def initialisation(u,K):
    # Step 0 : we choose K vectors among the data
    mu = np.zeros((K,d))
    I = np.random.randint(T,size=K)
    for j in range(K):
        mu[j] = u[I[j],:]
    return mu


def K_means(u,mu,K):
    nu = mu
    # Step 1
    M = np.sum((np.tensordot(u,np.ones(K),0)-np.transpose(np.tensordot(np.ones(T),mu,0),(0,2,1)))**2,1) # de taille (len(u),K) : matrice qui contient tous les ||u_i-mu_j||^2
    a = np.argmin(M,axis=1) # contient les indices j pour lesquels z[i,j]=1
    z = np.zeros((T,K))
    for (i,j) in enumerate(a): 
        z[i,j] = 1
    # Step 2 : on met à jour mu
    for j in range(K):
        if np.sum(z[:,j])>0:
            mu[j,:] = np.dot(z[:,j],u)/np.sum(z[:,j])
        else:
            mu[j,:] = np.zeros(d)
    if np.any(nu-mu) == False:
        return mu, z
    return K_means(u,mu,K)
    
    
def EM_DM2(T,K,d,u,pi,A,mu,cov):
    # E-Step : Calcul de tau 
    tau = np.zeros((T,K))
    for t in range(T):
        j_max=np.argmax([np.log(pi[j])-.5*np.log(np.linalg.det(cov[j]))-.5*np.dot(u[t]-mu[j],np.dot(np.linalg.inv(cov[j]),u[t]-mu[j])) for j in range(K)])
        tau[t,j_max]=1
    # M-step : 
    pi = np.sum(tau,axis=0)/T
    mu = np.zeros((K,d))
    cov = np.zeros((K,d,d))
    for j in range(K):
        mu[j] = np.sum(tau[:,j]*u.T,axis=1)/np.sum(tau[:,j])
        cov[j] = np.sum([tau[t,j]*np.tensordot(u[t]-mu[j],u[t]-mu[j],0) for t in range(T)],axis=0)/np.sum(tau[:,j])
    return pi, mu, cov


# Initialisation des paramètres mu, tau, pi, cov, A : 
mu = initialisation(u,K)
mu, tau = K_means(u,mu,K)
pi = np.sum(tau,axis=0)/T
A = (np.ones((K,K))+np.eye(K))/(K+1)
cov = np.zeros((K,d,d))
for j in range(K):
    cov[j] = np.sum([tau[t,j]*np.tensordot(u[t]-mu[j],u[t]-mu[j],0) for t in range(T)],axis=0)/np.sum(tau[:,j])


for nb_it in range(70): # nb itérations dans EM_DM2
    print(nb_it)
    pi, mu, cov = EM_DM2(T,K,d,u,pi,A,mu,cov)

# On stocke les variables pour calculer le log-likelihood
PI, MU, COV = pi, mu, cov
    
# Désormais, on trace les K ellipsoïdes de confiance
alpha = chi2.ppf(0.9, d) # quantile de la loi du chi2

# Pour définir l'intervalle à afficher
a1, b1 = np.min(u1), np.max(u1)
a2, b2 = np.min(u2), np.max(u2)
e, f = min(a1,a2), max(b1,b2) # pour bien délimiter l'affichage

nb_it = 1000 # nombre de points horizontal/vertical du maillage
XX, YY = np.meshgrid(np.linspace(e,f,nb_it), np.linspace(e,f,nb_it)) # maillage de l'espace

G=lambda x,A,b : A[0,0]*(x[0]-b[0])**2+2*A[0,1]*(x[0]-b[0])*(x[1]-b[1])+A[1,1]*(x[1]-b[1])**2<alpha

Pts=[]
for j in range(K):
    Pts.append(G([XX,YY], np.real(sqrtm(np.linalg.inv(cov[j]))), mu[j]))


# Script permettant de définir l'appartenance des données à chaque cluster
M = np.sum((np.tensordot(u,np.ones(K),0)-np.transpose(np.tensordot(np.ones(T),mu,0),(0,2,1)))**2,1)
z = np.zeros((T, K))
for t,i in enumerate(np.argmin(M, axis = 1)): 
    z[t,i] = 1 


# On trace les figures :
plt.figure(3)
plt.clf()
plt.axis([e,f,e,f])
MyColor=['g','darkslateblue','c','black']

for j in range(K): 
    xx1 = u1[np.nonzero(z[:,j])]
    xx2 = u2[np.nonzero(z[:,j])]
    plt.plot(xx1, xx2, '*', color = MyColor[j], label='data %s' %(j))
    plt.contour(XX, YY, Pts[j], 0, colors = MyColor[j], linewidths=2)
plt.plot(mu[:,0], mu[:,1], 'ro')
plt.title('EM - GMM') 
plt.legend()


  ## b) Implementation de l'algorithme EM_HMM
def EM(T,K,d,u,pi,A,mu,cov):
    # E-Step : Calcul de tau, prob_qq
    ''' tau[t,j] = p(q_t=j|u) et prob_qq[t,i,j]= p(q_t+1=i,q_t=j|u) '''
    ln_alpha, ln_beta = forward(T,K,u,pi,mu,cov,A), backward(T,K,u,pi,mu,cov,A)
    z = ln_alpha+ln_beta
    zz = np.tensordot(np.max(z, axis=1),np.ones(K),0)
    tau = np.exp(z-zz)
    tau = (tau.T/np.sum(tau,axis=1)).T
    a1 = np.tensordot(ln_alpha[:-1,:],np.ones(K),0)
    a2 = np.tensordot(ln_beta[1:,:],np.ones(K),0)
    a3 = np.tensordot(np.ones(T-1),A,0)
    a4 = np.transpose(np.tensordot(np.exp(func(u[1:],mu,cov)),np.ones(K),0),(1,2,0))
    a5 = a1+a2+a3+a4
    prob_qq = np.exp(a5-np.tensordot(np.max(a5,axis=(1,2)),np.ones((K,K)),0))
    prob_qq = (prob_qq.T/np.sum(prob_qq,axis=(1,2))).T
    # M-step : 
    # mise à jour de pi :
    pi = tau[0,:]
    pi *= 1/np.sum(pi)
    # mise à jour de A :
    A = np.sum(prob_qq, axis=0)
    A = (A.T/np.sum(A, axis=1)).T
    # mise à jour de mu et de cov :
    mu = np.zeros((K,d))
    cov = np.zeros((K,d,d))
    for j in range(K):
        mu[j] = np.sum(tau[:,j]*u.T,axis=1)/np.sum(tau[:,j])
        cov[j] = np.sum([tau[t,j]*np.tensordot(u[t]-mu[j],u[t]-mu[j],0) for t in range(T)],axis=0)/np.sum(tau[:,j])
    return pi, A, mu, cov, tau, prob_qq


for nb_it in range(10):
    print(nb_it)
    pi, A, mu, cov, tau, prob_qq = EM(T,K,d,u,pi,A,mu,cov)


# Désormais, on trace les K ellipsoïdes de confiance
Pts=[]
for j in range(K):
    Pts.append(G([XX,YY], np.real(sqrtm(np.linalg.inv(cov[j]))), mu[j]))


## Q4 - Inference algorithm for decoding
w = np.zeros((T, K))
w[0,:] = np.log(pi) + func(u[0],mu,cov)

for t in range(T-1):
    W = np.tensordot(np.ones(K), w[t,:], 0)
    w[t+1,:] = func(u[t+1,:],mu,cov)+np.max(W + np.log(A), axis = 1)


# backtracking
q = np.zeros(T)
q[T-1] = np.argmax(w[-1,:])

for t in range(T-2, 0, -1):
    i = q[t+1].astype(int)
    q[t] = np.argmax(w[t,:]+np.log(A[i,:]))

# On trace les figures
plt.figure(4)
plt.clf()
plt.axis([e,f,e,f])

for j in range(K): 
    xx1 = u1[np.where(q == j)]
    xx2 = u2[np.where(q == j)]
    plt.plot(xx1, xx2, '*', color = MyColor[j], label='data %s' %(j))
    plt.contour(XX, YY, Pts[j], 0, colors = MyColor[j], linewidths=2)
plt.plot(mu[:,0], mu[:,1], 'ro')
plt.title('EM - HMM') 
plt.legend()


## Q5 - Comparaison des log-likelihood
def log_likelihood_GMM(u,pi,mu,cov):
    T = len(u)
    # Script permettant de définir l'appartenance des données à chaque cluster
    M = np.sum((np.tensordot(u,np.ones(K),0)-np.transpose(np.tensordot(np.ones(T),mu,0),(0,2,1)))**2,1)
    z = np.argmin(M, axis = 1)
    log_likelihood = 0
    for t in range(T):
        log_likelihood += np.log(pi[z[t]])+func(u[t],mu,cov)[z[t]]
    return log_likelihood


def log_likelihood_HMM(u,pi, A, mu, cov):
    T = len(u)
    # Script permettant de définir l'appartenance des données à chaque cluster
    M = np.sum((np.tensordot(u,np.ones(K),0)-np.transpose(np.tensordot(np.ones(T),mu,0),(0,2,1)))**2,1)
    q = np.argmin(M, axis = 1)
    log_likelihood = np.log(pi[q[0]])
    for t in range(0, T - 1):
        log_likelihood += np.log(A[q[t],q[t + 1]])+func(u[t],mu,cov)[q[t]]
    return log_likelihood


# liste des log-likelihood pour les données d'entraînements ainsi que les données tests : 
log_likelihood_GMM = [log_likelihood_GMM(u_train,PI,MU,COV), log_likelihood_GMM(u_test,PI,MU,COV)]
log_likelihood_HMM = [log_likelihood_HMM(u_train,pi, A, mu, cov), log_likelihood_HMM(u_test,pi, A, mu, cov)]


print('log-likelihood de EM - GMM :', log_likelihood_GMM,'\nlog-likelihood de EM - HMM :', log_likelihood_HMM)
