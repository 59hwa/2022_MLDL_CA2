# Coding Assignment 2: K-means and GMM clustering.

## Introduction

The second coding assignment asks you to implement K-means clustering and EM algorithm for GMM.

We provide the code consisting of several Python files, which you will need to read and understand in order to complete the assignment.

**Note**: we will use `Python 3.x` for the project. 

## Deadline
June 3, 2022 11:59PM KST (*One day delay is permitted with linear scale score deduction. (20% per day)*)

### Submission checklist
* Push your code to [our github classroom page's CA2 section](https://classroom.github.com/a/MZou3O_s)
* Submit your report to [GEL](https://gel.gist.ac.kr)

---
## Preparation

### Installing prerequisites

The prerequisite usually refers to the necessary library that your code can run with. They are also known as `dependency`. To install the prerequisite, simply type in the shell prompt (not in a python interpreter) the following:

```
$ pip install -r requirements.txt
```

---
## Files

**Files you'll edit:**

* `datasets.py`: Data provider. 
* `kmeans.py`: You need to modify this file to implement a k-means clustering model.
* `gmm.py`: You need to modify this file to implement a GMM model.
* `util.py`: A bunch of utility functions!
* `test.py`: A testing code! I will run this code to run your models.


---
## What to submit
**Push to your github classroom** 

- All of the python files listed above (under "Files you'll edit"). 
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).

---
### Note
**Academic dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

---
##  K-means clustering (40%)

Write a program in python that performs K-means clustering. Run your own k-means clustering algorithm on toy data sets (data0.mat, data2.mat, data2.mat). Each data set contains two 2-dimensional data points, x an y. 

`REPORT1`: Run your k-means algorithms and describe what you found. Try your k-means with at least 3 different values of k (e.g., k=2,3,5). 

 Plot an error function with respect to the number of iterations. (see below for an example)
 
 ![error function plot example](https://github.com/gistmldl/ca2/blob/8164e54fcadeb0f66db681ca8f434a54353f96bd/images/kmeans_error.png)
 
 Draw your clustering results for each dataset. (see below for an example)
 
 ![visualization example](https://github.com/gistmldl/ca2/blob/8164e54fcadeb0f66db681ca8f434a54353f96bd/images/kmeans_plot.png)

---
##  EM algorithm for GMM (50%)

Write a program in python that performs GMM clustering. Run your own GMM clustering algorithm on toy data sets (data0.mat, data2.mat, data2.mat). Each data set contains two 2-dimensional data points, x an y. 

`REPORT2`: Run your GMM algorithms and describe what you found. Try your GMM with at least 3 different values of k (e.g., k=2,3,5). 

 Plot log-likelihood with respect to the number of iterations. (see below for an example)
 
 ![log-likelihood plot example](https://github.com/gistmldl/ca2/blob/8164e54fcadeb0f66db681ca8f434a54353f96bd/images/gmm_loglike.png)
 
 Draw your clustering results for each dataset. (see below for an example)
 
 ![visualization example](https://github.com/gistmldl/ca2/blob/8164e54fcadeb0f66db681ca8f434a54353f96bd/images/gmm_plot.png)
 
---
##  Discussion (10%)

`REPORT3`: What are the main practical differences between the two? Describe what you found with the toy datasets.
Let the number of clusters K = 2. How sensitive are the solutions to the starting points (try a number of them). 

---
##  Extra Credit (+30%)

Image segementation with k-means and GMM clustering.
Try your k-means and GMM with at least 3 different values of k (e.g., k=2,3,5) on image segmentation dataset (tiger.jpg, plane.jpg)

 ![tiger.jpg](https://github.com/gistmldl/ca2/blob/46a8b84363443f87e90bb0c1f0432434d9df19ff/images/tiger.jpg)
 ![plane.jpg](https://github.com/gistmldl/ca2/blob/46a8b84363443f87e90bb0c1f0432434d9df19ff/images/plane.jpg)
 

`REPORT4`: Visualize your resutls and describe what you found.

