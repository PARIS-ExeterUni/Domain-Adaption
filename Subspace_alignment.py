# Imports
import numpy as np
from sklearn.decomposition.pca import PCA
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Set seed for reproduceability
np.random.seed(41)

# Create the random data
X = np.random.multivariate_normal([0,0], [[2,30],[2,3]], (10000,39))[:,:,0]
#Create labels st classes = 0 or 1
y = (np.sum(X[:,12:16], 1) > 0)+0.0

# Split the data into the training and testing data sets
X_S = X[:5000]
X_T = X[5000:]
y_S = y[:5000]
y_T = y[5000:]

#unalign the test data so we can realign using Subspace alignment, dot prudct with shuffled identify is equalivent to random rotations
i = np.identity(39)
np.random.shuffle(i)
X_T = np.dot(X_T,i)

def pipeline(align, X_S, X_T, y_S):
	
	# Intitlaise objects
	Norm = Normalizer()
	Random_Forest = RandomForestClassifier(250,random_state = 42)
	SVM = SVC(random_state = 42, gamma = 'scale')

	# Change these values depending on the specifics of the data
	subspace_dimension = 3
		
	# Normalise the data
	X_S = Norm.fit_transform(X_S)
	X_T = Norm.fit_transform(X_T)
	
	# Create the PCA
	pca_train = PCA(n_components = subspace_dimension, random_state = 42)
	pca_test = PCA(n_components = subspace_dimension, random_state = 42)
	
	# Create the PCA components to reduce to subspace
	P_S = np.transpose(pca_train.fit(X_S).components_)
	P_T = np.transpose(pca_test.fit(X_T).components_)
	
	# In both cases:reduce source to subspace and reduce target to subspace
	# If using SA then rotate target data into source allignment
	if align == True:
		X_S_A = np.matmul(X_S, P_S)
		X_T_A = np.matmul(X_T,np.matmul(P_T, np.matmul(np.transpose(P_T), P_S)))
	else:
		X_S_A = np.matmul(X_S, P_S)
		X_T_A = np.matmul(X_T, P_S)
	
	# Train the classifiers on the source data
	Random_Forest.fit(X_S_A, y_S)
	SVM.fit(X_S_A, y_S)
	
	# Predict the labels
	Random_Forest_pred = Random_Forest.predict(X_T_A)
	SVM_pred = SVM.predict(X_T_A)
	return(np.array([Random_Forest_pred, SVM_pred]))

#Evaluate
RF_unaligned = pipeline(False, X_S, X_T, y_S)[0]
SVM_unaligned = pipeline(False, X_S, X_T, y_S)[1]

RF_aligned = pipeline(True, X_S, X_T, y_S)[0]
SVM_aligned = pipeline(True, X_S, X_T, y_S)[1]

print("Random forest scores")
print("Unalligned: " + str(accuracy_score(y_T, RF_unaligned)))
print("Alligned: " + str(accuracy_score(y_T, RF_aligned)))

print("SVM scores")
print("Unalligned: "+str(accuracy_score(y_T, SVM_unaligned)))
print("Alligned: "+str(accuracy_score(y_T, SVM_aligned)))
