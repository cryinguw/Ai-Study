손글씨를 분류

목차

1. 모듈 import하기
2. 데이터 준비
3. 데이터 이해
3. train, test 데이터 분리
4. 모델 학습
    1. Decision Tree
    2. Random Forest
    3. SVM
    4. SGD Classifier
    5.  Logistic Regression
5. 모델 평가

# 손글씨 분류

## 모듈 import하기


```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
```

## 데이터 준비


```python
digits = load_digits()
digits_data = digits.data
digits_label = digits.target
```


```python
print(dir(digits))
```

    ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']



```python
digits.keys()
```




    dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR'])



## 데이터 이해


```python
import matplotlib.pyplot as plt

for i in range(10):
    plt.subplot(2, 5, i+1) 
    plt.imshow(digits.data[i].reshape(8, 8), cmap='gray') 
    plt.axis('off')
plt.show()
```


    
![png](output_9_0.png)
    


## train, test 데이터 분리


```python
X_train, X_test, y_train, y_test = train_test_split(digits_data, 
                                                    digits_label, 
                                                    test_size=0.2, 
                                                    random_state=7) 
```

## 모델 학습

### A. Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier 

decision_tree = DecisionTreeClassifier(random_state=32)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.98      0.99        43
               1       0.81      0.81      0.81        42
               2       0.79      0.82      0.80        40
               3       0.79      0.91      0.85        34
               4       0.83      0.95      0.89        37
               5       0.90      0.96      0.93        28
               6       0.84      0.93      0.88        28
               7       0.96      0.82      0.89        33
               8       0.88      0.65      0.75        43
               9       0.78      0.78      0.78        32
    
        accuracy                           0.86       360
       macro avg       0.86      0.86      0.86       360
    weighted avg       0.86      0.86      0.85       360
    



```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt="d")
```




    <AxesSubplot:>




    
![png](output_15_1.png)
    


### B. Random Forest


```python
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(random_state=32) 
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.98      0.99        43
               1       0.93      1.00      0.97        42
               2       1.00      1.00      1.00        40
               3       1.00      1.00      1.00        34
               4       0.93      1.00      0.96        37
               5       0.90      0.96      0.93        28
               6       1.00      0.96      0.98        28
               7       0.94      0.97      0.96        33
               8       1.00      0.84      0.91        43
               9       0.94      0.94      0.94        32
    
        accuracy                           0.96       360
       macro avg       0.96      0.96      0.96       360
    weighted avg       0.97      0.96      0.96       360
    



```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt="d")
```




    <AxesSubplot:>




    
![png](output_18_1.png)
    


### C. SVM


```python
from sklearn import svm 

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        43
               1       0.95      1.00      0.98        42
               2       1.00      1.00      1.00        40
               3       1.00      1.00      1.00        34
               4       1.00      1.00      1.00        37
               5       0.93      1.00      0.97        28
               6       1.00      1.00      1.00        28
               7       1.00      1.00      1.00        33
               8       1.00      0.93      0.96        43
               9       1.00      0.97      0.98        32
    
        accuracy                           0.99       360
       macro avg       0.99      0.99      0.99       360
    weighted avg       0.99      0.99      0.99       360
    



```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt="d")
```




    <AxesSubplot:>




    
![png](output_21_1.png)
    


### D. SGD Classifier


```python
from sklearn.linear_model import SGDClassifier
sgd_model = SGDClassifier() 

sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        43
               1       0.91      0.74      0.82        42
               2       0.98      1.00      0.99        40
               3       0.89      0.94      0.91        34
               4       1.00      0.95      0.97        37
               5       0.96      0.96      0.96        28
               6       0.96      0.93      0.95        28
               7       0.94      0.97      0.96        33
               8       0.85      0.91      0.88        43
               9       0.86      0.97      0.91        32
    
        accuracy                           0.93       360
       macro avg       0.94      0.94      0.93       360
    weighted avg       0.93      0.93      0.93       360
    



```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt="d")
```




    <AxesSubplot:>




    
![png](output_24_1.png)
    


### E. Logistic Regression


```python
from sklearn.linear_model import LogisticRegression 
logistic_model = LogisticRegression() 

logistic_model.fit(X_train, y_train) 
y_pred = logistic_model.predict(X_test)

print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        43
               1       0.95      0.95      0.95        42
               2       0.98      1.00      0.99        40
               3       0.94      0.97      0.96        34
               4       0.97      1.00      0.99        37
               5       0.82      0.96      0.89        28
               6       1.00      0.96      0.98        28
               7       0.97      0.97      0.97        33
               8       0.92      0.81      0.86        43
               9       0.97      0.91      0.94        32
    
        accuracy                           0.95       360
       macro avg       0.95      0.95      0.95       360
    weighted avg       0.95      0.95      0.95       360
    


    /opt/conda/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True,fmt="d")
```




    <AxesSubplot:>




    
![png](output_27_1.png)
    



```python

```


```python

```
