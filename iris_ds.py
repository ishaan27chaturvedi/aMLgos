import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"
names = ['sl','sw','pl','pw','class']


#read data
data = pd.read_csv(url, names=names)


#Understanding data
print("Sample of data. Sepal Length, Sepal Width, Petal Length, Petal Width and class")
data.head()

print("Description of data")
data.describe()

print("Correlation of data columns, note how sl, pw and pl all correlate to each other well; while sw does not.")
data.corr()


#Visualizing data
print("Histogram of the data, notice how sw is the odd one out")
data.hist()
plt.show()

print("Kernel Density Estimation plot of the data")
data.plot(kind = 'kde')
plt.show()

print("Box plot of the data")
data.plot(kind = 'box')
plt.show()

print("Area plot of the data")
data.plot(kind = 'area')
plt.show()


#Visualizing Correlation of data
print("Scatter plot of pl and pw. The color intensity increases with sl. Note their correlation.")
data.plot.scatter(x='pl', y='pw', c='sl')
plt.show()

print("Hexbin plot of pl and pw. The color intensity increases with sl. Note their correlation.")
data.plot.hexbin(x='pl', y='pw', C='sl',reduce_C_function=np.max, gridsize=30)
plt.show()

print("Scatter Matrix")
pd.scatter_matrix(data, alpha=2,diagonal='kde')
plt.show()

from pandas.plotting import andrews_curves
print("Andrews curves")
andrews_curves(data,'class')
plt.show()

from pandas.plotting import parallel_coordinates
print("parallel coordinates")
parallel_coordinates(data, 'class')
plt.show()

from pandas.plotting import radviz
print("radviz. Note:Depending on which class that sample belongs it will be colored differently")
radviz(data, 'class')
plt.show()


#Preprocessing preparation of data
array = data.values
X = array[:,0:4]
Y = array[:,4]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])


#Resampling data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=7)			


#Algorithm evaluation with resampling methods
from sklearn.model_selection import cross_val_score

from sklearn import svm
clf = svm.SVR(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)                           

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
scores1 = cross_val_score(model, X_train, y_train, cv=kfold)
scores1
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

scores2 = cross_val_score(clf, X_test, y_test, cv=5)
scores2
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))


#Algorithm evaluation metrics
scores1 = cross_val_score(model, X_train, y_train, cv=kfold, scoring= 'neg_log_loss') #logloss
scores1
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

scores1 = cross_val_score(model, X_train, y_train, cv=kfold, scoring= 'accuracy') #accuracy
scores1
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))


#Model comparison and selection of algorithm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier

models = []
models.append(('LR', LogisticRegression()))
models.append(('PAC', PassiveAggressiveClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

