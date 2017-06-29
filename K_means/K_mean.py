import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
style.use('ggplot')

X = np.array([[1,2],
			 [1.5,1.8],
			 [5,8],
			 [8,8],
			 [1,0.6],
			 [9,11]])

clf = KMeans(n_clusters=2)
clf.fit(X)

centroid = clf.cluster_centers_
lable = clf.labels_

color = ["g.","r.","c.","y."]
for i in range(len(X)):
	plt.plot(X[i][0],X[i][1],color[lable[i]],markersize = 10)
plt.scatter(centroid[:,0],centroid[:,1],marker="x",s=50,linewidths=5,zorder=10)
plt.show()