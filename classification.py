from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
mnist = fetch_openml('mnist_784',version=1)
mnist.keys()
X,y = mnist["data"],mnist["target"]
X.shape
y.shape
some_digit= X[0]
some_digit_image=some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")
plt.show()
y[0]
y=y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# Training as binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)
sgd_clf.predict([some_digit])
# performance measures
# Alternative cross-validation
skfolds= StratifiedKFold(n_splits=3,random_state=42)
for train_index, test_index in skfolds.split(X_train,y_train_5):
    clone_clf= clone(sgd_clf)
    X_train_folds=X_train[train_index]
    y_train_folds=y_train_5[train_index]
    X_test_fold=X_train[test_index]
    y_test_fold=y_train_5[test_index]
    clone_clf.fit(X_train_folds,y_train_folds)
    y_pred=clone_clf.predict(X_test_fold)
    n_correct= sum(y_pred==y_test_fold)
    print(n_correct / len(y_pred))
# Standard cross-validation
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")    
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)
never_5_clf= Never5Classifier()  
cross_val_score(never_5_clf,X_train,y_train_5,cv=3,scoring="accuracy")  
# Confusion Matrix
y_train_pred= cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)    
confusion_matrix(y_train_5,y_train_pred)
# Precision and recall calculation
precision_score(y_train_5,y_train_pred)
recall_score(y_train_5,y_train_pred)
f1_score(y_train_5,y_train_pred)
# Trade-off calculation
y_scores = sgd_clf.decision_function([some_digit])
y_scores = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)
def plot_precision_recall_vs_threshold(precisions,recalls,thresholds):
    plt.plot(thresholds,precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds,recalls[:-1],"g--",label="Recall")
    [...]
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()    
threshold_90_precision = thresholds[np.argmax(precisions>=0.90)]
y_train_pred_90 = (y_scores>= threshold_90_precision)
precision_score(y_train_5,y_train_pred_90)
recall_score(y_train_5,y_train_pred_90)
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# ROC Curve
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1], [0,1], 'k--')
    [...]
plot_roc_curve(fpr,tpr)
plt.show()    
roc_auc_score(y_train_5,y_scores)
# Comparison of Randomforest with SGDClassifier via ROC
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
roc_auc_score(y_train_5,y_scores_forest)
# Multiclass classification
svm_clf = SVC()
svm_clf.fit(X_train,y_train)
svm_clf.predict([some_digit])
some_digit_scores = svm_clf.decision_function([some_digit])
np.argmax(some_digit_scores)
# Multiclass classification one vs rest classifier
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train,y_train)
ovr_clf.predict([some_digit])
sgd_clf.fit(X_train,y_train)
sgd_clf.predict([some_digit])
sgd_clf.decision_function([some_digit])
cross_val_score(sgd_clf,X_train,y_train,cv=3,scoring="accuracy")
scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf,X_train_scaled,y_train,cv=3,scoring="accuracy")
# Error analysis
y_train_pred= cross_val_predict(sgd_clf,X_train_scaled,y_train, cv=3)
conf_mx = confusion_matrix(y_train,y_train_pred)
plt.matshow(conf_mx,cmap=plt.cm.gray)
row_sums = conf_mx.sum(axis=1,keepdims=True)
norm_conf_mx= conf_mx/row_sums
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx,cmap=plt.cm.gray)
#Multilabel classification
y_train_large= (y_train>=7)
y_train_odd= (y_train % 2 ==1)
y_multilabel = np.c_[y_train_large,y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)
knn_clf.predict([some_digit])
y_train_knn_pred = cross_val_predict(knn_clf,X_train, y_multilabel, cv=3)
f1_score(y_multilabel,y_train_knn_pred, average='macro')
# Multioutput classification
noise = np.random.randint(0,100, (len(X_train),784))
X_train_mod = X_train+noise    
noise = np.random.randint(0, 100,(len(X_test),784))
X_test_mod = X_test+noise
y_train_mod = X_train
y_test_mod = X_test
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit =  knn_clf.predict([X_test_mod[0]])


