from IPython.display import display

import matplotlib as plt
import matplotlib.pyplot as plt
import mglearn
#
############################# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
# Clustering
from sklearn.mixture import GMM

# Models
# example of training a final classification model
from sklearn.datasets.samples_generator import make_blobs
# generate 2d classification dataset
#
#X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=1)
#X, y = make_blobs(random_state=42)
#
# fit final model
#model = LogisticRegression()
#model = MLPClassifier(solver='lbfgs', random_state=0)
#model = KNeighborsClassifier(n_neighbors= 3)
model = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1)
#model = GaussianNB()

#model = LinearSVC()
#model = SVC(kernel='rbf', C=10, gamma=0.1)

#model = GMM(n_components=5) # Clustering
model.fit(X, y)
#
# new instances where we do not know the answer
#
#Xnew, _ = make_blobs(n_samples=13, centers=2, n_features=2, random_state=1)
Xnew, _ = make_blobs(n_samples=13, centers=5, n_features=2, random_state=1)
#Xnew, _ = make_blobs(n_samples=18, random_state=42)
#
# Classifications
# make a prediction (Classification)
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

# Probability Predictions
# make a prediction (Probability)
ynew = model.predict_proba(Xnew)
# show the inputs and predicted probabilities
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

# Decision Boundry
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions
#
plot_decision_regions(X=X, y=y, clf=model, legend=2)
plt.show()
#######
# NEW #
#######
X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=1)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "r^")
    plt.plot(X[:, 0][y==3], X[:, 1][y==3], "y^")
    plt.plot(X[:, 0][y==4], X[:, 1][y==4], "ms")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-12.5, 1.5, -12.5, 7.5])
plt.show()
#######
# NEW #
#######
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "ms")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-12.5, 1.5, -12.5, 7.5])
plt.show()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(X, y)

import numpy as np

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

plot_predictions(polynomial_svm_clf, [-12.5, 1.5, -12.5, 7.5])
plot_dataset(X, y, [-12.5, 1.5, -12.5, 7.5])

plt.show()
