### PLASSIER Vincent : Master M2 MVA 2017/2018 - Graphical models - HWK 2


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.linalg import sqrtm

plt.ion()
plt.show()


# on charge les données d'entraînement
X_train=np.loadtxt('D:\Cours M2_MvA\Probabilistic Graphical Models\DM2\classification_data_HWK2\EMGaussian.data')


# on charge les données de test
X_test=np.loadtxt('D:\Cours M2_MvA\Probabilistic Graphical Models\DM2\classification_data_HWK2\EMGaussian.test')


X_train,X_test=X_train.T,X_test.T
X=X_train
X1,X2=X


K=4
p,n=np.shape(X)



## 1 - Implementation K_means


def initialisation(x,K):
    # Step 0 : we choose K vectors among the data
    mu=np.zeros((K,p))
    I=np.random.randint(n,size=K)
    for k in range(K):
        mu[k]=x[:,I[k]]
    return mu


def K_means(x,mu,K):
    nu=mu
    #Step 1
    A=np.sum((np.tensordot(x,np.ones(K),0)-np.transpose(np.tensordot(np.ones(n),mu,0),(2,0,1)))**2,0) # de taille (len(x),K) : matrice qui contient tous les ||x_i-mu_k||^2
    a=np.argmin(A,axis=1).reshape(n,1) # contient les indices k pour lesquels z[i,k]=1
    z=np.zeros((n,K))
    for (i,k) in enumerate(a): 
        z[i,k]=1 
    # Step 2 : on met à jour mu
    for k in range(K):
        if np.sum(z[:,k])>0:
            mu[k,:]=np.dot(x,z[:,k])/np.sum(z[:,k])
        else:
            mu[k,:]=np.zeros(p)
    if np.any(nu-mu)==False:
        return mu,z
    return K_means(x,mu,K)


''' Remarque : dans l'algorithme, on a évité les boucles pythons car celui-ci les supportent mal. C'est pourquoi on utilise des 'np.tensordot', 'np.argmin', et qu'on définit une fonction réccursive '''

''' Step 1 : Alternative :
for i in range(n): 
    q=0
    a=np.sum((x[:,i]-mu[0,:])**2)
    for k in range(K):
        b=np.sum((x[:,i]-mu[k,:])**2)
        if b<a:
            q=k
            a=b
    z[i,q]=1 '''


mu=initialisation(X,K)
mu,z=K_means(X,mu,K)

# Distorsion du modèle
J=np.sum(z*(np.tensordot(X,np.ones(K),0)-np.transpose(np.tensordot(np.ones(n),mu,0),(2,0,1)))**2)
print('distorsion:', J)


# Pour définir l'intervalle à afficher
a,b=np.min(X1),np.max(X1)
c,d=np.min(X2),np.max(X2)

# On trace les figures
plt.figure(1)
plt.clf()
plt.axis([a,b,c,d])
MyMarkers=['H','s','D','ro']
MyColor=['g','darkslateblue','c','black']
for j in range(K):
    xx1=X1[np.nonzero(X1*z[:,j])]
    xx2=X2[np.nonzero(X2*z[:,j])]
    plt.plot(xx1,xx2,'*',color=MyColor[j],label='data %s' %(j))
    plt.plot(mu[j,0],mu[j,1],MyMarkers[j],color=MyColor[j],markersize=18,label='centre %s' %(j))
plt.title('Illustration du K_means') 
plt.legend()


## 2 - Gaussian mixture model isotropic : Cov~Id

# Initialisation des paramètres mu, tau, pi, cov
mu=initialisation(X,K)
mu,tau=K_means(X,mu,K)
pi=np.sum(tau,axis=0)/n
cov=np.zeros((K,p,p))
for j in range(K):
    cov[j]=np.sum(tau[:,j]*np.sum((X-np.tensordot(mu[j],np.ones(n),0))**2,axis=0))/(p*np.sum(tau[:,j]))*np.eye(p)



def EM_isotropic(x,pi,mu,K,cov):
    # E-Step : Calcul de tau 
    tau=np.zeros((n,K))
    for i in range(n):
        j_max=np.argmax([np.log(pi[j])-.5*np.log(np.linalg.det(cov[j]))-.5*np.dot(x[:,i]-mu[j],np.dot(np.linalg.inv(cov[j]),x[:,i]-mu[j])) for j in range(K)])
        tau[i,j_max]=1
    # M-step : mise à jour de pi, mu, cov
    pi=np.sum(tau,axis=0)/n
    mu=np.zeros((K,p))
    cov=np.zeros((K,p,p))
    for j in range(K):
        mu[j,:]=np.sum(x*tau[:,j],axis=1)/np.sum(tau[:,j])
        cov[j]=np.sum(tau[:,j]*np.sum((x-np.tensordot(mu[j],np.ones(n),0))**2,axis=0))/(p*np.sum(tau[:,j]))*np.eye(p)
    return pi,mu,cov



for nb_it in range(10): # nb itérations dans EM_isotropic
    pi,mu,cov=EM_isotropic(X,pi,mu,K,cov)
    

# Désormais, on trace les K ellipsoïdes de confiance
alpha=chi2.ppf(0.9, p) # quantile de la loi du chi2
e,f=min(a,c),max(b,d) # pour bien délimiter l'affichage
nb_it=1000 # nombre de points horizontal/vertical du maillage
XX,YY=np.meshgrid(np.linspace(e,f,nb_it),np.linspace(e,f,nb_it)) # maillage de l'espace

G=lambda x,A,b : A[0,0]*(x[0]-b[0])**2+2*A[0,1]*(x[0]-b[0])*(x[1]-b[1])+A[1,1]*(x[1]-b[1])**2<alpha

Pts=[]
for j in range(K):
    Pts.append(G([XX,YY],np.real(sqrtm(np.linalg.inv(cov[j]))),mu[j]))



# Script permettant de définir l'appartenance des données à chaque cluster
A=np.sum((np.tensordot(X,np.ones(K),0)-np.transpose(np.tensordot(np.ones(n),mu,0),(2,0,1)))**2,0)
g=np.arange(n).reshape(n,1)
h=np.argmin(A,axis=1).reshape(n,1)
m=np.concatenate((g, h),axis=1) # contient tous les indices (i,k) pour lesquels z[i,k]=1
z=np.zeros((n,K))
for (i,k) in m: 
    z[i,k]=1 



# On trace les figures
plt.figure(2)
plt.clf()
plt.axis([e,f,e,f])
for j in range(K): 
    xx1=X1[np.nonzero(X1*z[:,j])]
    xx2=X2[np.nonzero(X2*z[:,j])]
    plt.plot(xx1,xx2,'*',color=MyColor[j],label='data %s' %(j))
    plt.contour(XX,YY,Pts[j],0,colors=MyColor[j],linewidths=2)
plt.plot(mu[:,0],mu[:,1],'ro')
plt.title('Gaussian mixture model isotropic') 
plt.legend()


## 3 - Gaussian mixture model standard

# Initialisation des paramètres mu, tau, pi, cov
mu=initialisation(X,K)
mu,tau=K_means(X,mu,K)
pi=np.sum(tau,axis=0)/n
cov=np.zeros((K,p,p))
for j in range(K):
    cov[j]=np.sum([np.array([tau[i,j]*np.tensordot(X[:,i]-mu[j],X[:,i]-mu[j],0)])  for i in range(n)],axis=0)/np.sum(tau[:,j])



def EM_general(x,pi,mu,K,cov):
    # E-Step : Calcul de tau 
    tau=np.zeros((n,K))
    for i in range(n):
        j_max=np.argmax([np.log(pi[j])-.5*np.log(np.linalg.det(cov[j]))-.5*np.dot(x[:,i]-mu[j],np.dot(np.linalg.inv(cov[j]),x[:,i]-mu[j])) for j in range(K)])
        tau[i,j_max]=1
    # M-step : 
    pi=np.sum(tau,axis=0)/n
    mu=np.zeros((K,p))
    cov=np.zeros((K,p,p))
    for j in range(K):
        mu[j,:]=np.sum(x*tau[:,j],axis=1)/np.sum(tau[:,j])
        cov[j]=np.sum([tau[i,j]*np.tensordot(x[:,i]-mu[j],x[:,i]-mu[j],0) for i in range(n)],axis=0)/np.sum(tau[:,j])
    return pi,mu,cov



for nb_it in range(10): # nb itérations dans EM_general
    pi,mu,cov=EM_general(X,pi,mu,K,cov)


# Désormais, on trace les K ellipsoïdes de confiance
Pts=[]
for j in range(K):
    Pts.append(G([XX,YY],np.linalg.inv(np.real(sqrtm(cov[j]))),mu[j]))



# Script permettant de définir l'appartenance des données à chaque cluster
A=np.sum((np.tensordot(X,np.ones(K),0)-np.transpose(np.tensordot(np.ones(n),mu,0),(2,0,1)))**2,0)
g=np.arange(n).reshape(n,1)
h=np.argmin(A,axis=1).reshape(n,1)
m=np.concatenate((g, h),axis=1) # contient tous les indices (i,k) pour lesquels z[i,k]=1
z=np.zeros((n,K))
for (i,k) in m: 
    z[i,k]=1 
    
    

# On trace les figures
plt.figure(3)
plt.clf()
for j in range(K): 
    xx1=X1[np.nonzero(X1*z[:,j])]
    xx2=X2[np.nonzero(X2*z[:,j])]
    plt.plot(xx1,xx2,'*',color=MyColor[j],label='data %s' %(j))
    plt.contour(XX,YY,Pts[j],0,colors=MyColor[j],linewidths=2)
plt.plot(mu[:,0],mu[:,1],'ro')
plt.title('Gaussian mixture general model') 
plt.legend()


## 4 - Comparaison of the two models
# X=X_train # log-likelihood des données d'entraînement
# X=X_test # pour afficher le log-likelihood des données test


log_likelihood=0
for i in range(n):
    log_likelihood+=np.max([np.log(pi[j])-.5*np.log(np.linalg.det(cov[j]))-.5*np.dot(X[:,i]-mu[j],np.dot(np.linalg.inv(cov[j]),X[:,i]-mu[j])) for j in range(K)])-.5*p*np.log(2*np.pi)
    
print('le log-likelihood de ce modèle vaut:', log_likelihood)