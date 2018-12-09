import anfis
from membership import membershipfunction, mfDerivs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
import numpy as np
boston = load_boston()
X = boston.data[:,0:5]
Y = boston.target

''' Reduced Feature unsupervised learning'''
# https://kite.com/python/docs/sklearn.datasets.lfw.Bunch
# https://scikit-learn.org/stable/modules/feature_selection.html
# https://scikit-learn.org/stable/modules/unsupervised_reduction.html


pca = PCA(n_components = 3)
X_less = pca.fit_transform(X)
print (X_less.shape)
boston.data = X_less
X = X_less

print (X, Y)


#mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
#            [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]

# jalan i = 2, j = 6; error = 27275.000
# PCA component = 3; i = 3; j = 4; epochs = 5, error = 25804.000
# PCA component = 3; i = 3; j = 5; epochs = 5, error = 25948.000

def get_membership(dataset):
    mf = []
    for i in range(3):
        gauss = []
        for j in range(4):
            filtered = []
            for k, target in enumerate(dataset.target):
                #print(j, round(target,-1)/ 10)
                if (abs(round(target, -1)/10) == j):
                    filtered.append(dataset.data[k][i])
                    #print("ini lala",j,dataset.data[k][i])
            if (np.mean(filtered) != 0 and np.var(filtered) != 0):
                gauss.append(['gaussmf', {'mean': np.mean(filtered), 'sigma': np.var(filtered)}])
            #print("\n==")
        mf.append(gauss)
        #print("\n++++")
    return mf

mf = get_membership(boston)         
print (mf)


mfc = membershipfunction.MemFuncs(mf)

anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=10)

for i in range(len(Y)):
        print(Y[i], round(anf.fittedValues[i][0],6))

anf.plotErrors()
anf.plotResults()
