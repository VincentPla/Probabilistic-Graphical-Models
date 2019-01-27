### PLASSIER Vincent : Master M2 MVA 2017/2018 - Graphical models - HWK 1


import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.show()


X1,X2,Y_train=np.loadtxt('D:\Cours M2_MvA\Probabilistic Graphical Models\DM1\classification_data_HWK1\classificationA.train').T
X_train=np.array([X1,X2]) # on charge les données d'entraînement
X,Y=X_train,Y_train
n=len(Y)

X1_test,X2_test,Y_test=np.loadtxt('D:\Cours M2_MvA\Probabilistic Graphical Models\DM1\classification_data_HWK1\classificationA.test').T
X_test=np.array([X1_test,X2_test]) # on charge les données de test
n_test=len(Y_test)


## 1 - Generative model (LDA)


sigma = lambda x : 1/(1+np.exp(-x)) # logistique régression

# 1_b)
n1=np.sum(Y)

pi=n1/n
mu1=np.sum(X*Y,axis=1)/n1
mu0=np.sum((1-Y)*X,axis=1)/(n-n1)
Z=X-np.tensordot(mu1,Y,0)-np.tensordot(mu0,1-Y,0)
cov=1/n*np.dot(Z,np.transpose(Z))

A=np.linalg.inv(cov)

omega1=np.dot(A,mu1-mu0)
alpha1=-.5*np.dot(np.transpose(mu1-mu0),np.dot(A,mu1+mu0))+np.log(pi/(1-pi))


a,b=np.min(X1),np.max(X1)
c,d=np.min(X2),np.max(X2)
xx=np.linspace(a,b,10**4)
yy=-(omega1[0]*xx+alpha1)/omega1[1]  


plt.figure(1)
plt.clf()
plt.axis([a,b,c,d])
plt.plot(X1[np.nonzero(X1*(1-Y))],X2[np.nonzero(X2*(1-Y))],'*',label='Y=0')
plt.plot(X1[np.nonzero(X1*Y)],X2[np.nonzero(X1*Y)],'*',label='Y=1')
plt.plot(xx,yy,color='darkslateblue',label='p(y=1|x)=0.5')
plt.title('Generative model (LDA)') 
plt.legend()



## 2 - Logistic regression

def Grad_Hess(Z0):
    etha=sigma(np.dot(Z0[:-1],X)+Z0[-1])
    g=np.array(list(np.dot(X,Y-etha))+[np.sum(Y-etha)]) 
    H=np.zeros((3,3))
    H[:2,:2]=-np.dot(X,np.dot(np.diag(etha*(1-etha)),X.T))
    H[2,2]=-np.sum(etha*(1-etha))
    H[0,2]=-np.sum(X1*etha*(1-etha))
    H[1,2]=-np.sum(X2*etha*(1-etha))
    H[2,0]=H[0,2]
    H[2,1]=H[1,2]
    return g,H
    
def Newton(Z0,epsilon,j_max,error,j):
    g,H=Grad_Hess(Z0)
    if 0 in np.linalg.eigvals(H):
        H+=0.01*np.eye(3)
    d=-np.dot(np.linalg.inv(H),g)
    Z1=Z0+d
    error=np.linalg.norm(Z1-Z0)
    Z0=Z1
    j+=1
    if j>j_max or error<epsilon:
        return Z1
    return Newton(Z0,epsilon,j_max,error,j)
    

Z0=np.zeros(3)
epsilon,j_max=10**-15,10**4
error,j=1,0

Z1=Newton(Z0,epsilon,j_max,error,j)
omega2,alpha2=Z1[:2],Z1[-1]


yy=-(omega2[0]*xx+alpha2)/omega2[1]

plt.figure(2)
plt.clf()
plt.axis([a,b,c,d])
plt.plot(X1[np.nonzero(X1*(1-Y))],X2[np.nonzero(X2*(1-Y))],'*',label='Y=0')
plt.plot(X1[np.nonzero(X1*Y)],X2[np.nonzero(X1*Y)],'*',label='Y=1')
plt.plot(xx,yy,color='darkslateblue',label='p(y=1|x)=0.5')
plt.title('Logistic regression')
plt.legend()


## 3 - Linear regression

X_tilde=np.array(list(np.ones(n))+list(X1)+list(X2)).reshape(3,n)

XX=np.dot(np.linalg.inv(np.dot(X_tilde,X_tilde.T)),X_tilde)

B=np.dot(XX,Y)
omega3,alpha3=B[1:],B[0]

yy=(.5-alpha3-omega3[0]*xx)/omega3[1]

plt.figure(3)
plt.clf()
plt.axis([a,b,c,d])
plt.plot(X1[np.nonzero(X1*(1-Y))],X2[np.nonzero(X2*(1-Y))],'*',label='Y=0')
plt.plot(X1[np.nonzero(X1*Y)],X2[np.nonzero(X1*Y)],'*',label='Y=1')
plt.plot(xx,yy,color='darkslateblue',label='p(y=1|x)=0.5')
plt.title('Linear regression')
plt.legend()


## 4 - Performances of the different methods


  ## a) Linear classification
miscl_err_train1=np.sum(abs(Y_train-(np.dot(omega1,X_train)+alpha1>0)))/n
miscl_err_test1=np.sum(abs(Y_test-(np.dot(omega1,X_test)+alpha1>0)))/n_test
print('Linear classification : \n error train=%s, error test=%s' %(int(miscl_err_train1*1000)/1000,int(miscl_err_test1*1000)/1000))

  ## a) Logistic regression
miscl_err_train2=np.sum(abs(Y_train-(np.dot(omega2,X_train)+alpha2>0)))/n
miscl_err_test2=np.sum(abs(Y_test-(np.dot(omega2,X_test)+alpha2>0)))/n_test
print('Logistic regression :\n error train=%s, error test=%s' %(int(miscl_err_train2*1000)/1000,int(miscl_err_test2*1000)/1000))

''' On remarque que l'erreur était nulle sur les données d'entraînement mais grande sur les données de test, on a donc de l'overfitting'''

  ## a) Linear regression
miscl_err_train3=np.sum(abs(Y_train-(np.dot(omega3,X_train)+alpha3-.5>0)))/n
miscl_err_test3=np.sum(abs(Y_test-(np.dot(omega3,X_test)+alpha3-.5>0)))/n_test
print('Linear regression :\n error train=%s, error test=%s' %(int(miscl_err_train3*1000)/1000,int(miscl_err_test3*1000)/1000))


## 5 - QDA model

Z=X-np.tensordot(mu0,np.ones(n),0)
cov0=1/(n-n1)*np.sum([np.tensordot(Z.T[i],(Z*(1-Y))[:,i],0) for i in range(n)],axis=0)
Z=X-np.tensordot(mu1,np.ones(n),0)
cov1=1/n1*np.sum([np.tensordot(Z.T[i],(Z*Y)[:,i],0) for i in range(n)],axis=0)

M=np.linalg.inv(cov0)-np.linalg.inv(cov1)
omega=np.dot(np.linalg.inv(cov1),mu1)-np.dot(np.linalg.inv(cov0),mu0)
alpha=0.5*np.dot(mu0,np.dot(np.linalg.inv(cov0),mu0))-0.5*np.dot(mu1,np.dot(np.linalg.inv(cov1),mu1))+np.log(pi/(1-pi))+.5*np.log(np.linalg.det(cov0))-.5*np.log(np.linalg.det(cov1))

j=1000
XX,YY=np.meshgrid(np.linspace(a,b,j),np.linspace(c,d,j))

G=lambda x,A,b,c : 0.5*(A[0,0]*x[0]**2+2*A[0,1]*x[0]*x[1]+A[1,1]*x[1]**2)+b[0]*x[0]+b[1]*x[1]+c>0
points=G([XX,YY],M,omega,alpha)


plt.figure(4)
plt.clf()
plt.axis([a,b,c,d])
plt.contour(XX,YY,points,[0],colors='darkslateblue',linewidths=2)
plt.plot(X1[np.nonzero(X1*(1-Y))],X2[np.nonzero(X2*(1-Y))],'*',label='Y=0')
plt.plot(X1[np.nonzero(X1*Y)],X2[np.nonzero(X1*Y)],'*',label='Y=1')
plt.title('QDA model')
plt.legend()


  ## c) QDA model
miscl_err_train4=np.sum(abs(Y_train-(.5*np.sum(X_train*np.dot(M,X_train),0)+np.dot(omega,X_train)+alpha>0)))/n
miscl_err_test4=np.sum(abs(Y_test-(.5*np.sum(X_test*np.dot(M,X_test),0)+np.dot(omega,X_test)+alpha>0)))/n_test
print('QDA model :\n error train=%s, error test=%s' %(int(miscl_err_train4*1000)/1000,int(miscl_err_test4*1000)/1000))