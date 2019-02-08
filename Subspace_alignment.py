# X_S and X_T are Source/ Target data repectively
# Y_S is the Source labelling		

import numpy as np
from sklearn.decomposition.pca import PCA
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

Norm = Normalizer()
Random_Forest = RandomForestClassifier(250,random_state = 42)
SVM = SVC(random_state = 42)

# Normalise the data
X_S = Normalizer.fit_transform(X_S)
X_T = Normalizer.fit_transform(X_T)

# Create the PCA
pca_train = PCA(n_components = subspace_dimension, random_state = 42)
pca_test = PCA(n_components = subspace_dimension, random_state = 42)

# Create the PCA components to reduce to subspace
P_S = np.transpose(pca_train.fit(X_S).components_)
P_T = np.transpose(pca_test.fit(X_T).components_)

# In both cases:reduce source to subspace and reduce target to subspace
# If using SA then rotate target data into source allignment
if allign == True:
	X_S_A = np.matmul(X_S, P_S)
	X_T_A = np.matmul(X_T,np.matmul(P_T, np.matmul(np.transpose(P_T), P_S)))
else:
	X_S_A = np.matmul(X_S, P_S)
	X_T_A = np.matmul(X_T, P_S)

# Train the classifier on the source data
Random_Forest.fit(X_S_A, Y_S)
SVM.fit(X_S_A, Y_S)

# Predict the labels
Random_Forest_pred = Random_Forest.predict(X_T_A)
SVM_pred = SVM.predict(X_T_A)
