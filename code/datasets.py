from numpy import *
import numpy as np
import util


class Dataset:
    """
    X is a feature vector
    """
    data = None
    
    def __init__(self, filename):
      
        self.data = np.load(filename)

    def getDataset_cluster(self):
       
        self.x = self.data  

        return self.x
    
