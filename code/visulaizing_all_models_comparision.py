import train_test
import matplotlib.pyplot as plt

from lazypredict.Supervised import LazyClassifier

x_train=train_test.x_train
x_test=train_test.x_test
y_train=train_test.y_train
y_test=train_test.y_test

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)

model_used=['ExtraTreesClassifier','LGBMClassifier','LabelSpreading','XGBClassifier','RandomForestClassifier','BaggingClassifier','LabelPropagation','ExtraTreeClassifier','DecisionTreeClassifier','SVC','NuSVC','AdaBoostClassifier','LinearDiscriminantAnalysis','RidgeClassifier','RidgeClassifierCV','LogisticRegression','CalibratedClassifierCV','LinearSVC','BernoulliNB','NearestCentroid','GaussianNB','KNeighborsClassifier','SGDClassifier','QuadraticDiscriminantAnalysis','Perceptron','PassiveAggressiveClassifier','DummyClassifier']

def plot_graphs(value):
    # horizontal bar graph
    plt.barh(model_used,models[value])
    plt.yticks(fontsize=9)
    plt.xlabel(value)
    plt.ylabel('models')
    plt.title(f'models vs {value} horizontal bar graph')
    plt.show()
    # line graph
    plt.plot(model_used,models[value],marker='o',markerfacecolor='blue', markersize=5)
    plt.xticks(rotation='vertical',fontsize=9)
    plt.ylabel(value)
    plt.xlabel('models')
    plt.title(f'models vs {value} line graph')
    plt.show()

plot_graphs('Accuracy')
plot_graphs('Balanced Accuracy')
plot_graphs('ROC AUC')
plot_graphs('F1 Score')
plot_graphs('Time Taken')