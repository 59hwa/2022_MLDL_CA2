"""
A starting code for a Gaussian Mixture Model.
"""

from numpy import *
import numpy as np
import random
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal ##비교용


class GMM:
    """
    This class is for the GMM implementation.
    """
    K=None
    mu = None
    sigma = None
    p = None
    expected = None
    loglike = [1.0,0.0]
    data = np.empty((),float)
    T = 10000
    def __init__(self, K,data):
        """
        Initialize our internal state.
        """
        if(len(data[:,0])==3):
            self.T=10
        self.K=K
        
        self.data = data.astype(np.float128)
        self.mu = np.zeros((0,len(self.data[:,0])),float)
        if (len(self.data[:,0])==3): ##noramlize
            for i in range(3):
                self.data[i]= self.data[i]/255
        for i in range(K):
            if (len(self.data[:,0])==2):
                self.mu = np.append(self.mu,np.array([[random.uniform(np.min(self.data[0]),np.max(self.data[0])),\
                                                            random.uniform(np.min(self.data[1]),np.max(self.data[1]))]]),axis=0)
                
                # self.mu = np.append(self.mu,np.array([[np.min(self.data[0]),np.min(self.data[1])]]),axis=0)##case 1-1
                
                # self.mu = np.append(self.mu,np.array([[(np.min(self.data[0])+np.max(self.data[0]))/2\
                #                                         ,(np.min(self.data[1])+np.max(self.data[1]))/2]]),axis=0)##case 1-2
                    
                # self.mu = np.append(self.mu,np.array([[np.max(self.data[0]),np.max(self.data[1])]]),axis=0)##case 1-3
                
                # if(i==0):
                #     self.mu = np.append(self.mu,np.array([[np.min(self.data[0]),np.min(self.data[1])]]),axis=0)
                # else:
                #     self.mu = np.append(self.mu,np.array([[np.maax(self.data[0]),np.max(self.data[1])]]),axis=0) ##case2
                
                # if(i==1):
                #     self.mu = np.append(self.mu,np.array([[np.min(self.data[0]),random.uniform(np.min(self.data[1]),np.max(self.data[1]))]]),axis=0)
                # else:
                #     self.mu = np.append(self.mu,np.array([[np.max(self.data[0]),random.uniform(np.min(self.data[1]),np.max(self.data[1]))]]),axis=0)##case3
                
                # self.mu = np.append(self.mu,np.array([[np.min(self.data[0]),random.uniform(np.min(self.data[1]),np.max(self.data[1]))]]),axis=0) ##case4-1
                
                # self.mu = np.append(self.mu,np.array([[np.max(self.data[0]),random.uniform(np.min(self.data[1]),np.max(self.data[1]))]]),axis=0) ##case4-2
                    
            elif(len(self.data[:,0])==3):
                self.mu = np.append(self.mu,np.array([[random.uniform(0,1),\
                                                            random.uniform(0,1),\
                                                            random.uniform(0,1)]]),axis=0)
        self.sigma = np.zeros((self.K,len(self.data[:,0]),len(self.data[:,0])))
        for i in range(self.K):
            self.sigma[i] = np.identity(n=len(self.data[:,0]))
            
        self.p = np.ones(self.K)/self.K
        self.expected = np.zeros((self.K,len(self.data[0])))
        self.count = 0
        if (len(self.data[:,0])==2):
            ax = plt.subplot()
            plt.scatter(data[0],data[1],color = 'b')
            plt.scatter(self.mu[:,0], self.mu[:,1], color = 'red', marker = 'x')
            plt.title('EM algorithm for GMM Initiation')
            plt.show()

    def run(self, gaussian):
        """
        Perform clustering
        """
        self.count +=1
        
        
        ### E-step
        self.loglike[1] = self.loglike[0]
        self.loglike[0] = 0
        for i in range(self.K):
            for j in range(len(self.data[0])):
                self.expected[i][j] = self.p[i]*gaussian[i][j]
        
        for j in range(len(self.data[0])):
            if (len(self.data[:,0])==2):
                self.loglike[0] = self.loglike[0] + np.log(np.sum(self.expected[:,j])/
                                                       (2*np.pi**(len(self.data[0])/2))) ## 없어도 결과엔 상관 x image seg에선 생략하고 하자
            elif(len(self.data[:,0])==3):
                self.loglike[0] = self.loglike[0] + np.log(np.sum(self.expected[:,j]))
            self.expected[:,j] = self.expected[:,j]/np.sum(self.expected[:,j])
            
            
        ### M-step

        for i in range(self.K):
            temp1 = np.zeros((len(self.data[:,0])))
            temp2 = 0
           
            for j in range(len(self.data[0])):
                temp1 = temp1+ self.expected[i][j]*self.data[:,j] ## 한점으로 가는건 어쩔수 없음
                temp2 = temp2+ self.expected[i][j]
            
            self.p[i] = temp2
            self.mu[i] = temp1/self.p[i]
            
            temp3 = np.zeros((len(self.data[:,0]),len(self.data[:,0])))
            for j in range(len(self.data[0])):
                temp3 = temp3+ self.expected[i][j]*np.dot((self.data[:,j]-self.mu[i]).reshape(len(self.data[:,0]),-1),\
                                                    (self.data[:,j]-self.mu[i]).reshape(-1,len(self.data[:,0])))
                self.sigma[i] = temp3/self.p[i]
                
                
        ## graph 출력 2차원일떄만 3차원은 시간이 너무걸림
        if (len(self.data[:,0])==2):
            ax = plt.subplot()
            for i in range(len(self.data[0])):
                if np.argmax(self.expected[:,i]) == 0:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'r')
                elif np.argmax(self.expected[:,i]) == 1:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'g')
                elif np.argmax(self.expected[:,i]) == 2:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'b')
                elif np.argmax(self.expected[:,i]) == 3:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'y')
                elif np.argmax(self.expected[:,i]) == 4:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'm')
            plt.scatter(self.mu[:,0], self.mu[:,1], color = 'red', marker = 'x')
            self.draw(ax, n_std=2.0, facecolor='none')
            plt.title('EM algorithm for GMM Step '+ str(self.count))
            plt.show()
        
        
    def stopping_criteria(self, loglike, prev_loglike):
        """
        Compute convergence criteria
        """
        print(loglike)
        if (loglike == prev_loglike):
            print(loglike)
            print(prev_loglike)
            print("convergence")
            if (len(self.data[:,0])==2):
                ax = plt.subplot()
                for i in range(len(self.data[0])):
                    if np.argmax(self.expected[:,i]) == 0:
                        plt.scatter(self.data[0][i],self.data[1][i],color = 'r')
                    elif np.argmax(self.expected[:,i]) == 1:
                        plt.scatter(self.data[0][i],self.data[1][i],color = 'g')
                    elif np.argmax(self.expected[:,i]) == 2:
                        plt.scatter(self.data[0][i],self.data[1][i],color = 'b')
                    elif np.argmax(self.expected[:,i]) == 3:
                        plt.scatter(self.data[0][i],self.data[1][i],color = 'y')
                    elif np.argmax(self.expected[:,i]) == 4:
                        plt.scatter(self.data[0][i],self.data[1][i],color = 'm')
                plt.scatter(self.mu[:,0], self.mu[:,1], color = 'red', marker = 'x')
                self.draw(ax, n_std=2.0, facecolor='none')
                plt.title('EM algorithm for GMM Result')
                plt.show()
            return True
        else :
            return False

    def gaussian(self, X, mu, sig):
        """
        Compute probability using Gaussian distribution
        """            
        # print(np.dot(np.dot((X-mu).reshape(-1,len(self.data[:,0])),np.linalg.inv(sig)),\
        #                                     (X-mu).reshape(len(self.data[:,0]),-1)))  
                                           # segmentation에서는 값이 너무 큼>noramlization 필요
            
        return np.exp(-1/2*np.dot(np.dot((X-mu).reshape(-1,len(self.data[:,0])),np.linalg.inv(sig)),\
                                          (X-mu).reshape(len(self.data[:,0]),-1)))\
                                          *1/np.sqrt((np.linalg.det(sig)))

## 그래프 타원 그리기 https://github.com/mr-easy/GMM-EM-Python/blob/master/GMM.py 참조

    def plot_gaussian(self, mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        '''
        Utility function to plot one Gaussian from mean and covariance.
        '''
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor,
            **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def draw(self, ax, n_std=2.0, facecolor='none', **kwargs):
        '''
        Function to draw the Gaussians.
        Note: Only for two-dimensionl dataset
        '''
        if(len(self.data[:,0]) != 2):
            print("Drawing available only for 2D case.")
            return
        for i in range(self.K):
            self.plot_gaussian(self.mu[i], self.sigma[i], ax, n_std=n_std, edgecolor='r', **kwargs)


                
