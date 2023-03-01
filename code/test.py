
from numpy import *
import numpy as np
from gmm import GMM
from kmeans import Kmeans
from datasets import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


k = 2
datafile = 'data3.npy'
img_color = plt.imread('tiger.jpg')
img = img_color.reshape(-1,3).T
fig = plt.figure()
ax = fig.gca(projection ='3d')
ax.scatter(img[0],img[1],img[2])
plt.show()

fig,ax = plt.subplots(figsize = (7,7))
plt.imshow(img_color)
plt.show()

data = Dataset(datafile)
a=data.getDataset_cluster()
plt.scatter(a[0],a[1])
plt.show()


#%% kmeans clustering
def kmeans(K,data):
    count_kmeans=0
    model = Kmeans(K,data)
    Error = np.zeros((model.T))
    # model.__init__(K,data)
    dist = np.zeros((K,len(data[0])),float)
    while True:
        for i in range(K):
            for j in range(len(data[0])):
                dist[i][j]=model.calc_dist(model.center[i], model.data[:,j])

        model.run(dist)
        count_kmeans=count_kmeans+1
        model.iter = count_kmeans
        for j in range(len(data[0])):
            Error[count_kmeans] += pow(dist[model.assign[0][j]][j],2)
        print(count_kmeans)
        if (model.stopping_criteria(model.assign[0], model.assign[1]) or count_kmeans==model.T ):
            break
        
    return model.center,model.assign, Error, count_kmeans


center,assign, Error, count = kmeans(k,a)
plt.scatter(range(1,count+1),Error[1:count+1])
plt.title('Error function')
plt.show()

for i in range(len(a[0])):
    if assign[0,i] == 0:
        plt.scatter(a[0][i],a[1][i],color = 'r')
    elif assign[0,i] == 1:
        plt.scatter(a[0][i],a[1][i],color = 'g')
    elif assign[0,i] == 2:
        plt.scatter(a[0][i],a[1][i],color = 'b')
    elif assign[0,i] == 3:
        plt.scatter(a[0][i],a[1][i],color = 'y')
    elif assign[0,i] == 4:
        plt.scatter(a[0][i],a[1][i],color = 'm')
    


plt.scatter(center[:,0],center[:,1],marker='X')
plt.title("K-Means clustering Result (" +datafile+ ')')
plt.show()

#%% kmeans segmentation kmeans에서 T=10
def kmeans_seg(K,data):
    count_kmeans=0
    model = Kmeans(K,data)
    # model.__init__(K,data)
    dist = np.zeros((K,len(data[0])),float)
    while True:
        for i in range(K):
            for j in range(len(data[0])):
                dist[i][j]=model.calc_dist(model.center[i], model.data[:,j])
        model.run(dist)
        count_kmeans=count_kmeans+1
        print(count_kmeans)
        if (model.stopping_criteria(model.assign[0], model.assign[1]) or count_kmeans==model.T ):
            break
        
    return model.center,model.assign


center,assign = kmeans_seg(k,img)
fig = plt.figure()
ax = fig.gca(projection ='3d')

# for i in range(len(img[0])):
#     if assign[0,i] == 0:
#         ax.scatter(img[0][i],img[1][i],img[2][i],color = 'r')
#     elif assign[0,i] == 1:
#         ax.scatter(img[0][i],img[1][i],img[2][i],color = 'g')
#     elif assign[0,i] == 2:
#         ax.scatter(img[0][i],img[1][i],img[2][i],color = 'b') // 시간이 너무 오래 걸림

ax.scatter(center[:,0],center[:,1],center[:,2],marker='X')
plt.show()
img_seg = assign[0].reshape(321,481)
img_segre = np.zeros((321,481,3),int)
for i in range(len(img_seg[0])):
    for j in range(len(img_seg[:,0])):
        img_segre[j,i]= center[img_seg[j,i]%3].astype(np.int32)

plt.imshow(img_segre)

#%% gmm clustering



def gmm(K,data):
    count_gmm =0
    model = GMM(K,data)
    loglike = np.zeros((model.T))
    gauss = np.zeros((K,len(data[0])),float)
    # print(model.mu)
    while True:
        for i in range(K):
            for j in range(len(data[0])):
                gauss[i][j] = model.gaussian(data[:,j],model.mu[i],model.sigma[i])

        model.run(gauss)
        loglike[count_gmm] = model.loglike[0]
        count_gmm+=1
        print(count_gmm)
        if(count_gmm == model.T) or model.stopping_criteria(model.loglike[0], model.loglike[1]):
            break
        
        
        
    return model.mu, model.expected, model.sigma, loglike, count_gmm


center, expected, sigma, loglike, gmm_count = gmm(k,a)
plt.scatter(range(gmm_count), loglike[:gmm_count])
plt.plot(range(gmm_count), loglike[:gmm_count])
plt.title("GMM loglikelihood (" +datafile+ ')')
plt.show()

##직접 그려볼려고 했으나 잘 안됐다. 좀 작게 나옴 (eigenvalue에 문제가 있나?)

# evalue, evector = np.linalg.eig(sigma)
# fig,ax = plt.subplots(figsize = (7,7))
# for i in range(len(a[0])):
#     if np.argmax(expected[:,i]) == 0:
#         plt.scatter(a[0][i],a[1][i],color = 'r')
#     elif np.argmax(expected[:,i]) == 1:
#         plt.scatter(a[0][i],a[1][i],color = 'g')
#     elif np.argmax(expected[:,i]) == 2:
#         plt.scatter(a[0][i],a[1][i],color = 'b')
        
# # plt.scatter(a[0],a[1])

# ax.add_patch(Ellipse((center[0,0],center[0,1]),evalue[0,0]**(1/2),evalue[0,1]**(1/2),angle = 0 ,edgecolor = 'b', fill = False))
# ax.add_patch(Ellipse((center[1,0],center[1,1]),evalue[1,0]**(1/2),evalue[1,1]**(1/2),angle = 0 ,edgecolor = 'r', fill = False))

# plt.scatter(center[:,0],center[:,1],marker='X')
# plt.show()

#%% gmm segmentation T=10
def gmm_seg(K,data):
    count_gmm =0

    model = GMM(K,data)

    gauss = np.zeros((K,len(data[0])),float)

    while True:
        for i in range(K):
            for j in range(len(data[0])):
                gauss[i][j] = model.gaussian(model.data[:,j],model.mu[i],model.sigma[i])
        model.run(gauss)
        count_gmm+=1
        print(count_gmm)
        if(count_gmm == model.T) or model.stopping_criteria(model.loglike[0], model.loglike[1]):
            break
        
        
    return model.mu, model.expected
    # return model.mu

center, expected = gmm_seg(k,img)


fig = plt.figure()
ax = fig.gca(projection ='3d')
ax.scatter(center[:,0]*255,center[:,1]*255,center[:,2]*255,marker='X')
plt.show()

img_seg = (expected.T).reshape(321,481,k)
img_segre = np.zeros((321,481,3),int)
for i in range(len(img_seg[0])):
    for j in range(len(img_seg[:,0])):
        img_segre[j,i]= (center[np.argmax(img_seg[j,i,:])]*255).astype(np.int32)
        
plt.imshow(img_segre)



