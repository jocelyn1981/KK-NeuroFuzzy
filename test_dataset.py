import membershipsteven as mbpstev
import numpy as np

'''
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.target)
'''

'''
from sklearn.datasets import load_diabetes
dia = load_diabetes()
print(dia.data)
#print(dia.target)
print(dia.feature_names)
#print(dia.target_names)

'''
'''

dataset = load_iris()
x = dataset.data
y = dataset.target
print(dataset.target)
print(dataset.target_names)

mf = mbpstev.get_mf(dataset)
print(mf)
'''

# https://kite.com/python/docs/sklearn.datasets.lfw.Bunch
# https://scikit-learn.org/stable/modules/feature_selection.html
# https://scikit-learn.org/stable/modules/unsupervised_reduction.html

from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
wine = load_wine()
X = wine.data
Y = wine.target
pca = PCA(n_components = 4)
X_less = pca.fit_transform(X)
print (X_less.shape)

wine.data = X_less

def get_membership(dataset):
    mf = []
    for i in range(4): # feature_size
        gauss = []
        for j in range(3): # target_class
            filtered = []
            for k, target in enumerate(dataset.target):
                if (target == j):
                    filtered.append(dataset.data[k][i])
            gauss.append(['gaussmf', {'mean': np.mean(filtered), 'sigma': np.var(filtered)}])
        mf.append(gauss)

    return mf


mf = get_membership(wine)
#mf = mbpstev.get_mf(wine)
print (mf) 
