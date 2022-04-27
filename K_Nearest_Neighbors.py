import numpy as np 
from collections import Counter

def euclidean_distances(x1, x2):
	### This function is used to calculate the euclidean distance between two points"""
	return np.sqrt(np.sum((x1 - x2) ** 2))
	
class KNN:
	def __init__(self, k=3):
		# Initializing the variables. K is the number of neighbors used
		self.k = k
	
	def fit(self, X, y):
		# This method is used to fit the data. Basically just storing inputed data for later use.
		self.X_train = X
		self.y_train = y
	
	def predict(self, X):
		# This method is used to actually predict/classify what we are interested in.
		predictions = [self._predict(x) for x in X]
		return np.array(predictions)
	
	def _predict(self, X):
		#This is an helper function that helps to calculate distances between the test data we want to predict/classify and also helps to give us the nearest neighbour
		distances = [euclidean_distances(X, x) for x in self.X_train]
		k_indices = np.argsort(distances)[:self.k]
		k_nearest_labels = [self.y_train[i] for i in k_indices]
		
		stats = Counter(k_nearest_labels).most_common(1)
		return (stats[0][0])
		
   
