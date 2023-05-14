import pandas as pd
import warnings
import train_test
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import  accuracy_score
from lazypredict.Supervised import LazyClassifier
warnings.filterwarnings('ignore', 'Solver terminated early.*')

# importing training and testing data from train_test
x_train=train_test.x_train
x_test=train_test.x_test
y_train=train_test.y_train
y_test=train_test.y_test

# APPLYING VARIOUS MODELS

# --- Applying Logistic Regression ---
LRclassifier = LogisticRegression(max_iter=1000, random_state=1, solver='liblinear', penalty='l1')
LRclassifier.fit(x_train, y_train)

y_pred_LR = LRclassifier.predict(x_test)

# --- LR Accuracy ---
LRAcc = accuracy_score(y_pred_LR, y_test)
print('<--- Logistic Regression Accuracy: '+'{:.2f}%'.format(LRAcc*100)+'---> ')

# --- Applying KNN ---
KNNClassifier = KNeighborsClassifier(n_neighbors=3)
KNNClassifier.fit(x_train, y_train)

y_pred_KNN = KNNClassifier.predict(x_test)

# --- KNN Accuracy ---
KNNAcc = accuracy_score(y_pred_KNN, y_test)
print('<--- K-Nearest Neighbour Accuracy: '+'{:.2f}%'.format(KNNAcc*100)+' --->')

# --- Applying SVM ---
SVMclassifier = SVC(kernel='linear', max_iter=1000, C=10, probability=True)
SVMclassifier.fit(x_train, y_train)

y_pred_SVM = SVMclassifier.predict(x_test)

# --- SVM Accuracy ---
SVMAcc = accuracy_score(y_pred_SVM, y_test)
print('<--- Support Vector Machine Accuracy: '+'{:.2f}%'.format(SVMAcc*100)+' --->')

# --- Applying Gaussian NB ---
GNBclassifier = GaussianNB(var_smoothing=0.1)
GNBclassifier.fit(x_train, y_train)

y_pred_GNB = GNBclassifier.predict(x_test)

# --- GNB Accuracy ---
GNBAcc = accuracy_score(y_pred_GNB, y_test)
print('<--- Gaussian Naive Bayes Accuracy: '+'{:.2f}%'.format(GNBAcc*100)+' --->')

# --- Applying Decision Tree ---
DTCclassifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy', min_samples_split=5,
splitter='random', random_state=1)

DTCclassifier.fit(x_train, y_train)
y_pred_DTC = DTCclassifier.predict(x_test)

# --- Decision Tree Accuracy ---
DTCAcc = accuracy_score(y_pred_DTC, y_test)
print('<--- Decision Tree Accuracy: '+'{:.2f}%'.format(DTCAcc*100)+' --->')

# --- Applying Random Forest ---
RFclassifier = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=20, min_samples_split=15)

RFclassifier.fit(x_train, y_train)
y_pred_RF = RFclassifier.predict(x_test)

# --- Random Forest Accuracy ---
RFAcc = accuracy_score(y_pred_RF, y_test)
print('<--- Random Forest Accuracy: '+'{:.2f}%'.format(RFAcc*100)+' --->')

# --- Applying Gradient Boosting ---
GBclassifier = GradientBoostingClassifier(random_state=1, n_estimators=100, max_leaf_nodes=3, loss='exponential', 
min_samples_leaf=20)

GBclassifier.fit(x_train, y_train)
y_pred_GB = GBclassifier.predict(x_test)

# --- Gradient Boosting Accuracy ---
GBAcc = accuracy_score(y_pred_GB, y_test)
print('<--- Gradient Boosting Accuracy: '+'{:.2f}%'.format(GBAcc*100)+' --->')

# --- Create Accuracy Comparison Table ---
compare = pd.DataFrame({'Model': ['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector Machine','Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],'Accuracy': [LRAcc*100, KNNAcc*100, SVMAcc*100, GNBAcc*100, DTCAcc*100, RFAcc*100, GBAcc*100]})

# # --- Create Accuracy Comparison Table ---
compare=compare.sort_values(by='Accuracy', ascending=False)

print(compare)

plt.plot(['Logistic Regression', 'K-Nearest Neighbour', 'Support Vector Machine','Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],[LRAcc*100, KNNAcc*100, SVMAcc*100, GNBAcc*100, DTCAcc*100, RFAcc*100, GBAcc*100],marker='o',markerfacecolor='blue', markersize=5)
plt.xticks(rotation='vertical',fontsize=9)
plt.ylabel('Accuracy')
plt.xlabel('Models')
plt.title('models vs accuracy line graph')
plt.show()

# comapring all models accuracy, roc, f1 value, time taken
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)

with pd.ExcelWriter("compare_all_models.xlsx") as writer:
    models.to_excel(writer,sheet_name="Sheet1")

