training_acc=[]
test_acc=[]
neighbors=range(1,31)

for k in range(1,31):
    KNNClassifier=KNeighborsClassifier(n_neighbors=k)
    KNNClassifier.fit(x_train,y_train)
    training_acc.append(KNNClassifier.score(x_train,y_train))
    test_acc.append(KNNClassifier.score(x_test,y_test))

plt.plot(neighbors,training_acc,label="training accuracy")
plt.plot(neighbors,test_acc,label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Value of k")
plt.legend()
plt.show()