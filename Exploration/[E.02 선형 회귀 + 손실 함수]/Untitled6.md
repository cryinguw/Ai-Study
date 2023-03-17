# 당뇨병 수치 예측

sklearn에 있는 데이터 중 하나인 당뇨병 데이터를 가지고 회귀분석을 해본다.

목차

[1. 데이터 가져오기](#데이터-가져오기)

[2. train 데이터와 test 데이터로 분리](#train-데이터와-test-데이터로-분리)

[3. 모델 준비](#모델-준비)

[4. 손실함수 loss 정의](#손실함수-loss-정의)

[5. gradient 함수](#gradient-함수)

[6. 모델 학습](#모델-학습)

[7. test 데이터 성능 확인](#test-데이터-성능-확인)


## 데이터 가져오기


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

diabetes=load_diabetes()
```


```python
diabetes.keys()
```




    dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])



data와 target을 받아온다.


```python
df_X=diabetes.data
df_y=diabetes.target

print(df_X.shape)
print(df_y.shape)
```

    (442, 10)
    (442,)



```python
diabetes_df = pd.DataFrame(df_X, columns=diabetes.feature_names)
diabetes_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>437</th>
      <td>0.041708</td>
      <td>0.050680</td>
      <td>0.019662</td>
      <td>0.059744</td>
      <td>-0.005697</td>
      <td>-0.002566</td>
      <td>-0.028674</td>
      <td>-0.002592</td>
      <td>0.031193</td>
      <td>0.007207</td>
    </tr>
    <tr>
      <th>438</th>
      <td>-0.005515</td>
      <td>0.050680</td>
      <td>-0.015906</td>
      <td>-0.067642</td>
      <td>0.049341</td>
      <td>0.079165</td>
      <td>-0.028674</td>
      <td>0.034309</td>
      <td>-0.018118</td>
      <td>0.044485</td>
    </tr>
    <tr>
      <th>439</th>
      <td>0.041708</td>
      <td>0.050680</td>
      <td>-0.015906</td>
      <td>0.017282</td>
      <td>-0.037344</td>
      <td>-0.013840</td>
      <td>-0.024993</td>
      <td>-0.011080</td>
      <td>-0.046879</td>
      <td>0.015491</td>
    </tr>
    <tr>
      <th>440</th>
      <td>-0.045472</td>
      <td>-0.044642</td>
      <td>0.039062</td>
      <td>0.001215</td>
      <td>0.016318</td>
      <td>0.015283</td>
      <td>-0.028674</td>
      <td>0.026560</td>
      <td>0.044528</td>
      <td>-0.025930</td>
    </tr>
    <tr>
      <th>441</th>
      <td>-0.045472</td>
      <td>-0.044642</td>
      <td>-0.073030</td>
      <td>-0.081414</td>
      <td>0.083740</td>
      <td>0.027809</td>
      <td>0.173816</td>
      <td>-0.039493</td>
      <td>-0.004220</td>
      <td>0.003064</td>
    </tr>
  </tbody>
</table>
<p>442 rows × 10 columns</p>
</div>



나이가 소수점으로 표현되어 있는데 모든 특성이 -0.2에서 0.2사이에 분포하도록 조정

## train 데이터와 test 데이터로 분리


```python
df_X=diabetes.data

df_X
```




    array([[ 0.03807591,  0.05068012,  0.06169621, ..., -0.00259226,
             0.01990842, -0.01764613],
           [-0.00188202, -0.04464164, -0.05147406, ..., -0.03949338,
            -0.06832974, -0.09220405],
           [ 0.08529891,  0.05068012,  0.04445121, ..., -0.00259226,
             0.00286377, -0.02593034],
           ...,
           [ 0.04170844,  0.05068012, -0.01590626, ..., -0.01107952,
            -0.04687948,  0.01549073],
           [-0.04547248, -0.04464164,  0.03906215, ...,  0.02655962,
             0.04452837, -0.02593034],
           [-0.04547248, -0.04464164, -0.0730303 , ..., -0.03949338,
            -0.00421986,  0.00306441]])




```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

    (353, 10) (353,)
    (89, 10) (89,)


## 모델 준비


```python
W = np.random.rand(10)
b = np.random.rand()

print(W)
print(b)
```

    [0.28729098 0.01730835 0.18514486 0.987159   0.53912636 0.19438237
     0.2416875  0.38141055 0.63327959 0.93867489]
    0.5778334460934594


## 손실함수 loss 정의

손실함수는 MSE함수로 사용해본다.


```python
def model(df_X, W, b):
    predictions = 0
    for i in range(10):
        predictions += df_X[:, i] * W[i]
    predictions += b
    return predictions
```


```python
def MSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse
```


```python
def loss(df_X, W, b, df_y):
    predictions = model(df_X, W, b)
    L = MSE(predictions, df_y)
    return L
```

## gradient 함수


```python
def gradient(df_X, W, b, df_y):
    # N은 데이터 포인트의 개수
    N = len(df_y)
    
    # y_pred 준비
    y_pred = model(df_X, W, b)
    
    # 공식에 맞게 gradient 계산
    dW = 1/N * 2 * df_X.T.dot(y_pred - df_y)
        
    # b의 gradient 계산
    db = 2 * (y_pred - df_y).mean()
    return dW, db


dW, db = gradient(df_X, W, b, df_y)

print("dW:", dW)
print("db:", db)
```

    dW: [-1.37027278 -0.31231578 -4.28915424 -3.22508241 -1.54447354 -1.2675622
      2.88829422 -3.14516451 -4.13592183 -2.79238527]
    db: -303.1113014336049


## 모델 학습

학습률을 0.1로 설정하고

10000번 학습시켜본다.


```python
LEARNING_RATE = 0.01

losses = []

for i in range(1, 100001):
    dW, db = gradient(X_train, W, b, y_train)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)
    if i % 5000 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))
```

    Iteration 5000 : Loss 3915.8930
    Iteration 10000 : Loss 3345.0486
    Iteration 15000 : Loss 3108.1053
    Iteration 20000 : Loss 2977.2691
    Iteration 25000 : Loss 2897.2055
    Iteration 30000 : Loss 2846.5355
    Iteration 35000 : Loss 2813.9848
    Iteration 40000 : Loss 2792.8540
    Iteration 45000 : Loss 2779.0046
    Iteration 50000 : Loss 2769.8388
    Iteration 55000 : Loss 2763.7104
    Iteration 60000 : Loss 2759.5675
    Iteration 65000 : Loss 2756.7331
    Iteration 70000 : Loss 2754.7681
    Iteration 75000 : Loss 2753.3857
    Iteration 80000 : Loss 2752.3972
    Iteration 85000 : Loss 2751.6776
    Iteration 90000 : Loss 2751.1430
    Iteration 95000 : Loss 2750.7374
    Iteration 100000 : Loss 2750.4223



```python
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
```


    
![png](output_24_0.png)
    


4000번 학습부터 LOSS값이 2700에 수렴하는 것을 볼 수 있다.

loss값이 큰 이유는 상대적으로 x값들에 비해 y값이 매우 크기 때문.

## test 데이터 성능 확인


```python
prediction = model(X_test, W, b)
mse = loss(X_test, W, b, y_test)
mse
```




    3451.081763173368




```python
plt.scatter(X_test[:, 0], y_test)
plt.scatter(X_test[:, 0], prediction)
plt.show()
```


    
![png](output_28_0.png)
    

