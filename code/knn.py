# since got a better accuracy using KNN finding the optimal value of n_neighbors which increases the accuracy 
import train_test
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

x_train=train_test.x_train
y_train=train_test.y_train
x_test=train_test.x_test
y_test=train_test.y_test

training_acc=[]
test_acc=[]
neighbors=range(1,41)

for k in range(1,41):
    KNNClassifier=KNeighborsClassifier(n_neighbors=k)
    KNNClassifier.fit(x_train,y_train)
    training_acc.append(KNNClassifier.score(x_train,y_train))
    test_acc.append(KNNClassifier.score(x_test,y_test))
    
plt.plot(neighbors,training_acc)
plt.ylabel("Accuracy")
plt.xlabel("Value of k")
plt.title('Accuracy on training dataset')
plt.show()

plt.plot(neighbors,test_acc)
plt.ylabel("Accuracy")
plt.xlabel("Value of k")
plt.title('Accuracy on training dataset')
plt.show()

# Can see from the graph that the accuracy for test data is maximum for k=1
kNNClassifier=KNeighborsClassifier(n_neighbors=1)
kNNClassifier.fit(x_train,y_train)
y_pred_kNN=kNNClassifier.predict(x_test)
kNNAcc=accuracy_score(y_pred_kNN,y_test)
pickle.dump(kNNClassifier, open('test.pkl', 'wb'))
print(f'Optimal value of k is 1, where we get an accuracy of {kNNAcc*100}%')

total=y_test.size
correct=kNNAcc*total
wrong=total-correct
plt.bar(['Correct output','Wrong output'],[correct,wrong])
plt.ylabel('Total')
plt.xlabel('Output')
plt.title('Output Accuracy on testing dataset')
plt.show()

