from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree



#task 1
print("")
print("task1")
iris = load_iris()

X = iris.data
y = iris.target

print("Feature names:", iris.feature_names)
print("Class names:", iris.target_names)
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("First five samples:\n", X[:5])
print("First five labels:", y[:5])


#task2
print("")
print("task2")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predicted classes:", predictions)
print("Actual classes:", y_test)


#task 3
print("")
print("task3")
accuracy = accuracy_score(y_test, predictions)

print("Test accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot()
plt.show()

#task 4
print("")
print("task4")

for depth in [1, 2, 3, 5, None]:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train, train_predictions
    )

    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    print(
        f"Depth: {depth}, "
        f"Training accuracy: {train_accuracy:.3f}, "
        f"Test accuracy: {test_accuracy:.3f}"
    )


#task 5
plt.figure(figsize=(16, 8))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

plt.show()