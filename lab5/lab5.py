from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

#Task 1
iris = load_iris()

X = iris.data
y = iris.target

print("number of samples: ", X.shape[0])
print("number of features: ", X.shape[1])
print("flower species names: ", iris.target_names)

#Task 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("number of training samples: ", X_train.shape[0])
print("number of testing samples: ", X_test.shape[0])

#Task 3
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


for i in range(len(y_pred)):
    print("predicted species: ", iris.target_names[y_pred[i]] + ", actual species: ", iris.target_names[y_test[i]])


#Task 4
f1 = f1_score(y_test,y_pred,  average='macro')
print("f1 score: ", f1)
accuracy = accuracy_score(y_test,y_pred)
print("accuracy: ", accuracy)


#Task 5
depths = [1,3]
for i in depths:
    model = DecisionTreeClassifier(max_depth=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test,y_pred, average='macro')
    accuracy = accuracy_score(y_test,y_pred)
    print(" ")
    print("max depth: ", i)
    print("f1 score: ", f1)
    print("accuracy: ", accuracy)
    print(" ")


