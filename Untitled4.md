## 데이터 불러오기


```python
import numpy as np 
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import warnings  
warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv('high_diamond_ranked_10min.csv')

# KDA 추가하기 KDA = (kill + assist) / death

df['blueKDA'] = (df['blueKills'] + df['blueAssists'])/ df['blueDeaths']
df['redKDA'] = (df['redKills'] + df['redAssists'])/ df['redDeaths']

df
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
      <th>gameId</th>
      <th>blueWins</th>
      <th>blueWardsPlaced</th>
      <th>blueWardsDestroyed</th>
      <th>blueFirstBlood</th>
      <th>blueKills</th>
      <th>blueDeaths</th>
      <th>blueAssists</th>
      <th>blueEliteMonsters</th>
      <th>blueDragons</th>
      <th>...</th>
      <th>redAvgLevel</th>
      <th>redTotalExperience</th>
      <th>redTotalMinionsKilled</th>
      <th>redTotalJungleMinionsKilled</th>
      <th>redGoldDiff</th>
      <th>redExperienceDiff</th>
      <th>redCSPerMin</th>
      <th>redGoldPerMin</th>
      <th>blueKDA</th>
      <th>redKDA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4519157822</td>
      <td>0</td>
      <td>28</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>6</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6.8</td>
      <td>17047</td>
      <td>197</td>
      <td>55</td>
      <td>-643</td>
      <td>8</td>
      <td>19.7</td>
      <td>1656.7</td>
      <td>3.333333</td>
      <td>1.555556</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4523371949</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6.8</td>
      <td>17438</td>
      <td>240</td>
      <td>52</td>
      <td>2908</td>
      <td>1173</td>
      <td>24.0</td>
      <td>1762.0</td>
      <td>2.000000</td>
      <td>1.400000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4521474530</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>6.8</td>
      <td>17254</td>
      <td>203</td>
      <td>28</td>
      <td>1172</td>
      <td>1033</td>
      <td>20.3</td>
      <td>1728.5</td>
      <td>1.000000</td>
      <td>3.571429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4524384067</td>
      <td>0</td>
      <td>43</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>7.0</td>
      <td>17961</td>
      <td>235</td>
      <td>47</td>
      <td>1321</td>
      <td>7</td>
      <td>23.5</td>
      <td>1647.8</td>
      <td>1.800000</td>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4436033771</td>
      <td>0</td>
      <td>75</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7.0</td>
      <td>18313</td>
      <td>225</td>
      <td>67</td>
      <td>1004</td>
      <td>-230</td>
      <td>22.5</td>
      <td>1740.4</td>
      <td>2.000000</td>
      <td>2.166667</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>9874</th>
      <td>4527873286</td>
      <td>1</td>
      <td>17</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>6.8</td>
      <td>16498</td>
      <td>229</td>
      <td>34</td>
      <td>-2519</td>
      <td>-2469</td>
      <td>22.9</td>
      <td>1524.6</td>
      <td>3.000000</td>
      <td>1.571429</td>
    </tr>
    <tr>
      <th>9875</th>
      <td>4527797466</td>
      <td>1</td>
      <td>54</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>7.0</td>
      <td>18367</td>
      <td>206</td>
      <td>56</td>
      <td>-782</td>
      <td>-888</td>
      <td>20.6</td>
      <td>1545.6</td>
      <td>3.500000</td>
      <td>1.166667</td>
    </tr>
    <tr>
      <th>9876</th>
      <td>4527713716</td>
      <td>0</td>
      <td>23</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>7</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>7.4</td>
      <td>19909</td>
      <td>261</td>
      <td>60</td>
      <td>2416</td>
      <td>1877</td>
      <td>26.1</td>
      <td>1831.9</td>
      <td>1.571429</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>9877</th>
      <td>4527628313</td>
      <td>0</td>
      <td>14</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>7.2</td>
      <td>18314</td>
      <td>247</td>
      <td>40</td>
      <td>839</td>
      <td>1085</td>
      <td>24.7</td>
      <td>1529.8</td>
      <td>1.666667</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>9878</th>
      <td>4523772935</td>
      <td>1</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6.8</td>
      <td>17379</td>
      <td>201</td>
      <td>46</td>
      <td>-927</td>
      <td>58</td>
      <td>20.1</td>
      <td>1533.9</td>
      <td>1.833333</td>
      <td>1.666667</td>
    </tr>
  </tbody>
</table>
<p>9879 rows × 42 columns</p>
</div>




```python
display(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9879 entries, 0 to 9878
    Data columns (total 42 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   gameId                        9879 non-null   int64  
     1   blueWins                      9879 non-null   int64  
     2   blueWardsPlaced               9879 non-null   int64  
     3   blueWardsDestroyed            9879 non-null   int64  
     4   blueFirstBlood                9879 non-null   int64  
     5   blueKills                     9879 non-null   int64  
     6   blueDeaths                    9879 non-null   int64  
     7   blueAssists                   9879 non-null   int64  
     8   blueEliteMonsters             9879 non-null   int64  
     9   blueDragons                   9879 non-null   int64  
     10  blueHeralds                   9879 non-null   int64  
     11  blueTowersDestroyed           9879 non-null   int64  
     12  blueTotalGold                 9879 non-null   int64  
     13  blueAvgLevel                  9879 non-null   float64
     14  blueTotalExperience           9879 non-null   int64  
     15  blueTotalMinionsKilled        9879 non-null   int64  
     16  blueTotalJungleMinionsKilled  9879 non-null   int64  
     17  blueGoldDiff                  9879 non-null   int64  
     18  blueExperienceDiff            9879 non-null   int64  
     19  blueCSPerMin                  9879 non-null   float64
     20  blueGoldPerMin                9879 non-null   float64
     21  redWardsPlaced                9879 non-null   int64  
     22  redWardsDestroyed             9879 non-null   int64  
     23  redFirstBlood                 9879 non-null   int64  
     24  redKills                      9879 non-null   int64  
     25  redDeaths                     9879 non-null   int64  
     26  redAssists                    9879 non-null   int64  
     27  redEliteMonsters              9879 non-null   int64  
     28  redDragons                    9879 non-null   int64  
     29  redHeralds                    9879 non-null   int64  
     30  redTowersDestroyed            9879 non-null   int64  
     31  redTotalGold                  9879 non-null   int64  
     32  redAvgLevel                   9879 non-null   float64
     33  redTotalExperience            9879 non-null   int64  
     34  redTotalMinionsKilled         9879 non-null   int64  
     35  redTotalJungleMinionsKilled   9879 non-null   int64  
     36  redGoldDiff                   9879 non-null   int64  
     37  redExperienceDiff             9879 non-null   int64  
     38  redCSPerMin                   9879 non-null   float64
     39  redGoldPerMin                 9879 non-null   float64
     40  blueKDA                       9879 non-null   float64
     41  redKDA                        9879 non-null   float64
    dtypes: float64(8), int64(34)
    memory usage: 3.2 MB
    


    None



```python
for c in df.columns:
    print('{} : {}'.format(c, len(df.loc[pd.isnull(df[c]), c].values)))
```

    gameId : 0
    blueWins : 0
    blueWardsPlaced : 0
    blueWardsDestroyed : 0
    blueFirstBlood : 0
    blueKills : 0
    blueDeaths : 0
    blueAssists : 0
    blueEliteMonsters : 0
    blueDragons : 0
    blueHeralds : 0
    blueTowersDestroyed : 0
    blueTotalGold : 0
    blueAvgLevel : 0
    blueTotalExperience : 0
    blueTotalMinionsKilled : 0
    blueTotalJungleMinionsKilled : 0
    blueGoldDiff : 0
    blueExperienceDiff : 0
    blueCSPerMin : 0
    blueGoldPerMin : 0
    redWardsPlaced : 0
    redWardsDestroyed : 0
    redFirstBlood : 0
    redKills : 0
    redDeaths : 0
    redAssists : 0
    redEliteMonsters : 0
    redDragons : 0
    redHeralds : 0
    redTowersDestroyed : 0
    redTotalGold : 0
    redAvgLevel : 0
    redTotalExperience : 0
    redTotalMinionsKilled : 0
    redTotalJungleMinionsKilled : 0
    redGoldDiff : 0
    redExperienceDiff : 0
    redCSPerMin : 0
    redGoldPerMin : 0
    blueKDA : 0
    redKDA : 0
    

## 데이터 시각화


```python
blue_side_columns = []
for col in df.columns:
       if "blue" in col:
              blue_side_columns.append(col)
                
blue_data = df[blue_side_columns]
```


```python
plt.figure(figsize=(12, 10))
sns.heatmap(blue_data.drop("blueWins",axis=1).corr(),cmap='vlag', annot=True, fmt='.2f')
```




    <AxesSubplot:>




    
![png](output_7_1.png)
    



```python
plt.figure(figsize=(6,3))
plt.title('blueWins')
plt.ylabel('Amount of wins')
sns.countplot(df['blueWins'])
print(df['blueWins'].value_counts())
```

    0    4949
    1    4930
    Name: blueWins, dtype: int64
    


    
![png](output_8_1.png)
    


### 다이아 수준

와드설치, 와드제거, 분당 CS는 실력간의 차이가 크기 때문에 따로 비교해보았다.


```python
fig, ax = plt.subplots(ncols=2, figsize=(8,4))

sns.histplot(data=df, x="blueWardsPlaced", ax=ax[0], color="b")
ax[0].set_xlim(0,60)

sns.histplot(data=df, x="redWardsPlaced", ax=ax[1], color="r")
ax[1].set_xlim(0,60)
plt.tight_layout()
```


    
![png](output_11_0.png)
    



```python
print(f"블루가 이길 때 평균 와드 설치 개수는 {round(df[df['blueWins'] == 1]['blueWardsPlaced'].mean(),2)}개 이다.")
print(f"블루가 질 때 평균 와드 설치 개수는 {round(df[df['blueWins'] == 0]['blueWardsPlaced'].mean(),2)}개 이다.")
print(f"레드가 이길 때 평균 와드 설치 개수는 {round(df[df['blueWins'] == 0]['redWardsPlaced'].mean(),2)}개 이다.")
print(f"레드가 질 때 평균 와드 설치 개수는 {round(df[df['blueWins'] == 1]['redWardsPlaced'].mean(),2)}개 이다.")
```

    블루가 이길 때 평균 와드 설치 개수는 22.29개 이다.
    블루가 질 때 평균 와드 설치 개수는 22.29개 이다.
    레드가 이길 때 평균 와드 설치 개수는 22.8개 이다.
    레드가 질 때 평균 와드 설치 개수는 21.93개 이다.
    


```python
fig, ax = plt.subplots(ncols=2, figsize=(8,4))

sns.histplot(data=df, x="blueWardsDestroyed", ax=ax[0], color="b")
ax[0].set_xlim(0,20)

sns.histplot(data=df, x="redWardsDestroyed", ax=ax[1], color="r")
ax[1].set_xlim(0,20)
plt.tight_layout()
```


    
![png](output_13_0.png)
    



```python
print(f"블루가 이길 때 평균 와드 파괴 개수는 {round(df[df['blueWins'] == 1]['blueWardsDestroyed'].mean(),2)}개 이다.")
print(f"블루가 질 때 평균 와드 파괴 개수는 {round(df[df['blueWins'] == 0]['blueWardsDestroyed'].mean(),2)}개 이다.")
print(f"레드가 이길 때 평균 와드 파괴 개수는 {round(df[df['blueWins'] == 0]['redWardsDestroyed'].mean(),2)}개 이다.")
print(f"레드가 질 때 평균 와드 파괴 개수는 {round(df[df['blueWins'] == 1]['redWardsDestroyed'].mean(),2)}개 이다.")
```

    블루가 이길 때 평균 와드 파괴 개수는 2.92개 이다.
    블루가 질 때 평균 와드 파괴 개수는 2.73개 이다.
    레드가 이길 때 평균 와드 파괴 개수는 2.84개 이다.
    레드가 질 때 평균 와드 파괴 개수는 2.6개 이다.
    


```python
fig, ax = plt.subplots(ncols=2, figsize=(8,4))

sns.histplot(data=df, x="blueCSPerMin", ax=ax[0], color="b")
ax[0].set_xlim(5,35)

sns.histplot(data=df, x="redCSPerMin", ax=ax[1], color="r")
ax[1].set_xlim(5,35)
plt.tight_layout()
```


    
![png](output_15_0.png)
    



```python
print(f"블루가 이길 때 평균 분당 CS 개수는 {round(df[df['blueWins'] == 1]['blueCSPerMin'].mean(),2)}개 이다.")
print(f"블루가 질 때 평균 분당 CS 개수는 {round(df[df['blueWins'] == 0]['blueCSPerMin'].mean(),2)}개 이다.")
print(f"레드가 이길 때 평균 분당 CS 개수는 {round(df[df['blueWins'] == 0]['redCSPerMin'].mean(),2)}개 이다.")
print(f"레드가 질 때 평균 분당 CS 개수는 {round(df[df['blueWins'] == 1]['redCSPerMin'].mean(),2)}개 이다.")
```

    블루가 이길 때 평균 분당 CS 개수는 22.16개 이다.
    블루가 질 때 평균 분당 CS 개수는 21.18개 이다.
    레드가 이길 때 평균 분당 CS 개수는 22.2개 이다.
    레드가 질 때 평균 분당 CS 개수는 21.27개 이다.
    

### 승패에 영향 끼치는 요소들


```python
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))

sns.histplot(x ='blueGoldDiff',data = df, hue = 'blueWins',ax=axs[0][0])
sns.countplot(x = 'blueEliteMonsters', data = df, hue = 'blueWins', ax=axs[0][1])
sns.countplot(x = 'blueFirstBlood', data = df, hue = 'blueWins', ax=axs[1][0])
sns.countplot(x = 'blueTowersDestroyed', data = df, hue = 'blueWins', ax=axs[1][1])
```




    <AxesSubplot:xlabel='blueTowersDestroyed', ylabel='count'>




    
![png](output_18_1.png)
    



```python
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))

sns.histplot(x ='blueGoldDiff',data = df, hue = 'blueFirstBlood',ax=axs[0][0])
sns.scatterplot(x ='blueTotalGold', y = 'blueKills', data = df, hue = 'blueWins',ax=axs[0][1])
sns.scatterplot(x ='blueTotalGold', y = 'blueKDA', data = df, hue = 'blueWins',ax=axs[1][0])
sns.scatterplot(x='blueExperienceDiff', y='blueGoldDiff', hue='blueWins', data=df,ax=axs[1][1])
```




    <AxesSubplot:xlabel='blueExperienceDiff', ylabel='blueGoldDiff'>




    
![png](output_19_1.png)
    



```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

sns.histplot(x = 'blueGoldPerMin', data = df, hue = 'blueWins', ax=ax1)
sns.histplot(x = 'blueTotalJungleMinionsKilled', data = df, hue = 'blueWins', ax=ax2)
```




    <AxesSubplot:xlabel='blueTotalJungleMinionsKilled', ylabel='Count'>




    
![png](output_20_1.png)
    


## 가설

1. 정글이 겜의 판도를 좌지우지하는데 10분 동안 먹은 정글 몹을 통해 팀의 승패가 결정되는가?



