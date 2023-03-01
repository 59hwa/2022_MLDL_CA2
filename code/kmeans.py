"""
A starting code for a K-means algorithm.
"""

from numpy import *
import numpy as np
import random
import matplotlib.pyplot as plt

class Kmeans:
    """
    This class is for the K-means implementation.
    """
    data = None
    center = None
    T = 10000
    K = None
    assign = None
    
    def __init__(self, K,data):
        """
        Initialize our internal state.
        """
        ##v.1 just random
        if(len(data[:,0])==3):
            self.T=10
        self.K = K
        self.center = np.zeros((0,len(data[:,0])),float)
        self.data = data
        self.iter = 0
        for i in range(K):
            if (len(data[:,0])==2):
                self.center = np.append(self.center,np.array([[random.uniform(np.min(data[0]),np.max(data[0])),\
                                                            random.uniform(np.min(data[1]),np.max(data[1]))]]),axis=0)
                
                # self.center = np.append(self.center,np.array([[np.min(self.data[0]),np.min(self.data[1])]]),axis=0)##case 1-1
                
                # self.center = np.append(self.center,np.array([[(np.min(self.data[0])+np.max(self.data[0]))/2\
                #                                         ,(np.min(self.data[1])+np.max(self.data[1]))/2]]),axis=0)##case 1-2
                    
                # self.center = np.append(self.center,np.array([[np.max(self.data[0]),np.max(self.data[1])]]),axis=0)##case 1-3
                # if(i==0):
                #     self.center = np.append(self.center,np.array([[np.min(self.data[0]),np.min(self.data[1])]]),axis=0)
                # else:
                #     self.center = np.append(self.center,np.array([[np.max(self.data[0]),np.max(self.data[1])]]),axis=0) ##case2
                # if(i==1):
                #     self.center = np.append(self.center,np.array([[np.min(self.data[0]),\
                #                                                 random.uniform(np.min(data[1]),np.max(data[1]))]]),axis=0)
                # else:
                #     self.center = np.append(self.center,np.array([[np.max(self.data[0]),\
                                                                # random.uniform(np.min(data[1]),np.max(data[1]))]]),axis=0) ##case3
                
                # self.center = np.append(self.center,np.array([[np.min(self.data[0]),\
                #                                                 random.uniform(np.min(data[1]),np.max(data[1]))]]),axis=0)##case4-1
                # self.center = np.append(self.center,np.array([[np.max(self.data[0]),\
                #                                                 random.uniform(np.min(data[1]),np.max(data[1]))]]),axis=0)##case4-2
                        
                
                
            elif(len(data[:,0])==3):
                self.center = np.append(self.center,np.array([[random.uniform(np.min(data[0]),np.max(data[0])),\
                                                            random.uniform(np.min(data[1]),np.max(data[1])),\
                                                            random.uniform(np.min(data[2]),np.max(data[2]))]]),axis=0)
        self.assign = np.zeros((2,len(data[0])),int)
        if (len(data[:,0])==2):
            plt.scatter(self.data[0],self.data[1])
            plt.scatter(self.center[:,0],self.center[:,1],marker='X')
            plt.title("Kmeans clustering initiation")
            plt.show()
        ##v.2
        
    def run(self, dist):
        
        """
        Perform clustering 
        """

        ### Assignment step
        
        self.assign[1] = self.assign[0]
        
        for i in range(len(dist[0])):
            self.assign[0][i] = np.argmin(dist[:,i])

        ### Update step
        
        for i in range(self.K):
            count = 0
            temp= np.zeros((len(self.data[:,0])),float)
            for j in range(len(dist[0])):
                if self.assign[0][j] == i:
                    temp = [temp[k] + self.data[k,j] for k in range(len(temp))]
                    count+=1
          
            self.center[i] = [temp[k]*1/count for k in range(len(temp))]

        if (len(self.data[:,0])==2):
            for i in range(len(self.data[0])):
                if self.assign[0,i] == 0:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'r')
                elif self.assign[0,i] == 1:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'g')
                elif self.assign[0,i] == 2:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'b')
                elif self.assign[0,i] == 3:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'y')
                elif self.assign[0,i] ==4:
                    plt.scatter(self.data[0][i],self.data[1][i],color = 'm')
            plt.scatter(self.center[:,0],self.center[:,1],marker='X')
            plt.title("K-Means clustering Step " + str(self.iter+1))
            plt.show()
    def stopping_criteria(self, loglike, prev_loglike):
        
        """
        Compute convergence criteria    L이 증가하지 않음을 보이면 convergence 한다고 argue가능
        간단하게는 assign 이 안변함을 보이면 됨
        """
        if (loglike == prev_loglike).min():
            print("convergence")
            if (len(self.data[:,0])==2):
                for i in range(len(self.data[0])):
                   if self.assign[0,i] == 0:
                       plt.scatter(self.data[0][i],self.data[1][i],color = 'r')
                   elif self.assign[0,i] == 1:
                       plt.scatter(self.data[0][i],self.data[1][i],color = 'g')
                   elif self.assign[0,i] == 2:
                       plt.scatter(self.data[0][i],self.data[1][i],color = 'b')
                   elif self.assign[0,i] == 3:
                       plt.scatter(self.data[0][i],self.data[1][i],color = 'y')
                   elif self.assign[0,i] ==4:
                       plt.scatter(self.data[0][i],self.data[1][i],color = 'm')
                plt.scatter(self.center[:,0],self.center[:,1],marker='X')
                plt.title("K-Means clustering Step Result")
                plt.show()
            return True
        else :
            return False
       


    def calc_dist(self, X, Y):
        """
        Compute distance between two vectors
        """
        return np.sqrt(pow(X[0]-Y[0],2) +pow(X[1]-Y[1],2))
        



                
