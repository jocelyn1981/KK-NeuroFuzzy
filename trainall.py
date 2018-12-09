import anfis
from membership import membershipfunction, mfDerivs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import numpy as np

wine = load_wine()
X = wine.data
Y = wine.target

''' Reduced Feature unsupervised learning'''
# https://kite.com/python/docs/sklearn.datasets.lfw.Bunch
# https://scikit-learn.org/stable/modules/feature_selection.html
# https://scikit-learn.org/stable/modules/unsupervised_reduction.html


pca = PCA(n_components = 3)
X_less = pca.fit_transform(X)
print (X_less.shape)
wine.data = X_less
X = X_less


def get_membership(dataset):
    mf = []
    for i in range(3): # feature_size
        gauss = []
        for j, tname in enumerate(dataset.target_names): # target_class
            filtered = []
            for k, target in enumerate(dataset.target):
                if (target == j):
                    filtered.append(dataset.data[k][i])
            gauss.append(['gaussmf', {'mean': np.mean(filtered), 'sigma': np.var(filtered)}])
            # print("\n==")
        mf.append(gauss)
        # print("\n++++")
    return mf

mf = get_membership(wine)
print (mf)

mfc = membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs = 10)

# current error in PCA 2 features: 50
# current error in PCA 3 features: 38
# current error in PCA 4 features: 14

Y_predict = []

for i in range(len(Y)):
    predict_value = round(anf.fittedValues[Y[i]][0], 6)
    print ("Y test: " + str(Y[i]), "Y predicted: " + str(predict_value))


# print(classification_report(Y, Y_predict, target_names = wine.target_names))

anf.plotErrors()
anf.plotResults()

