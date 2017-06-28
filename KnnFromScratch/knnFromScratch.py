import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_feature = [5,7]


def k_nearest_neighbors(data, predict, k=3):
	if len(data)>=k:
		warnings.warn('NOT A GOOD CHOICE')

	distance = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)- np.array(predict))
			distance.append([euclidean_distance,group])

	votes = [i[1] for i in sorted(distance)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

result = k_nearest_neighbors(dataset,new_feature)
print(result)

for i in dataset:
	for ii in dataset[i]:
		plt.scatter(ii[0],ii[1],s=100,color=i)


plt.scatter(new_feature[0],new_feature[1],s=100)
plt.show()

