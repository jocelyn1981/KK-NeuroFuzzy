from operator import truediv

import pandas as pd
import anfis
#from anfis import anfis
#from anfis import membership
from membership import membershipfunction
from membership import mfDerivs
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import membershipsteven as mbpstev

dataset = load_iris()
x = dataset.data
y = dataset.target


# for train, test in kf.split(X, Y):
df = pd.DataFrame(x)
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
y_test = y_test.tolist()
print ("Length: ",len(y_test))

mf = mbpstev.get_mf(dataset)
print (mf)

mfc = membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(x_train, y_train, mfc)
anf.trainHybridJangOffLine(epochs=10)
y_predicted = []

for i in range(len(y_test)):
    res = round(anf.fittedValues[y_test[i]][0],1)
    print ("Y test: " + str(y_test[i]), "Y predicted: " + str(res))
    if abs(res-0) < abs(res -1) < abs(res -2):
       y_predicted.append(0)
    elif abs(res-0) > abs(res -1) < abs(res -2):
       y_predicted.append(1)
    elif abs(res-0) > abs(res-1) > abs(res-2):
       y_predicted.append(2)

trupred = 0
print (y_test)
print (y_predicted)


#check accuracy
for i in range(len(y_predicted)):
    if y_predicted[i] == y_test[i]:
        trupred +=1

print ("Sum of TruePrediction: ",trupred)
print (truediv(trupred,len(y_test))*100, "%")

print(classification_report(y_test, y_predicted))
anf.plotResults()

