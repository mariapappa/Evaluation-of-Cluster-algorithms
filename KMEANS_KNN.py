#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
#import kNN
from sklearn.neighbors import KNeighborsClassifier
#from matplotlib import rc
from matplotlib.colors import ListedColormap
#import iris dataset as iris
from sklearn.datasets import load_iris
iris = load_iris()
#iris is a bunch object containing both data and target. 
#copy them as X and y, respectively
X = iris.data
y = iris.target
print(X.shape, y.shape)
print(type(X), type(y))
print(iris.feature_names)

print(iris.target_names)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)
#calculate kNN performace for k= 1 to 99
perform_all=[]
for k in range(1,100):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    perform_all.append(clf.score(X_test, y_test))
#    print(k, clf.score(X_test, y_test))
#plot k vs kNN performance 
plt.plot(perform_all, '-og')
plt.title('k vs Accuracy')
plt.xlabel('k' , fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.ylim((0.4,1.05))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
################################################################################
#reloading data and target into X, and y
X = iris.data
y = iris.target

index1 = 2
index2 = 3

var1 = X[:, index1]
var2 = X[:, index2]
X_2var = np.column_stack((var1,var2))
#import k-means 
from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import pairwise_distances_argmin


#k=3, use default values for others
clf = KMeans(n_clusters=3)
#fitting
clf.fit(X_2var)
cent = clf.cluster_centers_
w = 0.02

xm = np.arange(min(var1)-0.5, max(var1)+0.5, w)
ym = np.arange(min(var2)-0.5, max(var2)+0.5, w)

xx, yy = np.meshgrid(xm,ym)

zz = clf.predict(np.c_[xx.ravel(),yy.ravel()])
zz=zz.reshape(xx.shape)
cmap_light = ListedColormap([ 'pink','lightgreen', 'lightblue'])
# Create color maps
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.pcolormesh(xx, yy, zz, cmap=cmap_light)
plt.scatter(var1,var2, c=y,cmap=cmap_bold)
xx.shape
plt.xlim(xx[0,0], xx[0,-1])
plt.ylim(yy[0,0], yy[-1,0])
plt.xlabel(iris.feature_names[index1], fontsize=15)
plt.ylabel(iris.feature_names[index2], fontsize=15)
plt.scatter(cent[:,0],cent[:,1], marker='x',s=200, lw=3,c='k')
plt.show()
#Repeat kNN calculation, but this time with only 
#two features (columns 2 and 3)
X_2var = np.column_stack((var1,var2))

#split training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_2var, y, test_size=0.25, random_state=19)
perform_2var=[]

#kNN performance for different k values (1-99)
for k in range(1,100):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    perform_2var.append(clf.score(X_test, y_test))
#compare the two graphs
plt.plot(perform_2var, '-og', perform_all, '-^r')
plt.title('k vs Accuracy')
plt.xlabel('k' , fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.ylim((0.4,1.05))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Two Features','Four Features'], loc=3)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
#reloading data and target into X, and y
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)#, random_state=44)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
clf.coef_
y_pred=clf.predict(X_test)
plt.scatter(X_test[:,index1], X_test[:, index2], c=y_test, cmap=cmap_light, marker='s', s=100)
plt.scatter(X_test[:,index1], X_test[:, index2], c=y_pred, cmap=cmap_bold, edgecolors='white')
plt.xlabel(iris.feature_names[index1], fontsize=15)
plt.ylabel(iris.feature_names[index2], fontsize=15)
plt.title('Logistic Regression')
plt.show()
