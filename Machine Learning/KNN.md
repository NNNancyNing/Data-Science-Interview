## General Conception
- **Conception**: 

    To predict the class or value of a new observation by looking at k nearest observations in the training set. 
    
    For classification problem, the class label is the highest frequency label of k neighbors; 
    
    For regression problem, the predicted value is the mean or median of k neighbors. 
    
    Distance between observations: usually Eucledian distance, (or other distance metrics)
    
- **Drawbacks**: computationally expensive, especially when dealing with large datasets or high-dimensional data.

- **Selection of K**: Through cross validation, choose k with the smallest SS on test set(Not on training set, or will always choose k=1 so that RSS always=0)

- **Bias-variance trade-off of K**: When k is very large, then we take average of too many points, so the variance is very small but bias is very big. 


## Example Code
```
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

#Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

#Create hyperparameter Grid
k_values = list(range(1, 31))

#Execute Grid Search
cv_scores = []
for k in k_values:
  knn = KNeighborsClassifier(n_neighbors = k)
  scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
  cv_scores.append(scores.mean())

#Evaluate performance & Choose the best hyperparameter
best_k = k_values[cv_scores.index(max(cv_scores))]

#Refit model with best hyperparameter, apply on test set to obtain result
knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#Evaluate performance on test set
accuracy = knn.score(X_test, y_test)
cm = confusion_matrix(y_test, y_pred)

```
