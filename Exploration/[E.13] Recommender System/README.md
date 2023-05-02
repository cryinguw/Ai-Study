# Movielens 영화 추천


```python
import pandas as pd
import os
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import os
import numpy as np
```

## 데이터 가져오기


```python
rating_file_path=os.getenv('HOME') + '/aiffel/recommendata_iu/data/ml-1m/ratings.dat'
ratings_cols = ['user_id', 'movie_id', 'ratings', 'timestamp']
ratings = pd.read_csv(rating_file_path, sep='::', names=ratings_cols, engine='python', encoding = "ISO-8859-1")
orginal_data_size = len(ratings)
ratings.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>ratings</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1193</td>
      <td>5</td>
      <td>978300760</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>661</td>
      <td>3</td>
      <td>978302109</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>914</td>
      <td>3</td>
      <td>978301968</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3408</td>
      <td>4</td>
      <td>978300275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2355</td>
      <td>5</td>
      <td>978824291</td>
    </tr>
  </tbody>
</table>
</div>



## 데이터 전처리


```python
# 3점 이상만 남기기
ratings = ratings[ratings['ratings']>=3]
filtered_data_size = len(ratings)

print(f'orginal_data_size: {orginal_data_size}, filtered_data_size: {filtered_data_size}')
print(f'Ratio of Remaining Data is {filtered_data_size / orginal_data_size:.2%}')
```

    orginal_data_size: 1000209, filtered_data_size: 836478
    Ratio of Remaining Data is 83.63%



```python
# ratings 컬럼의 이름을 counts로 바꾸기
ratings.rename(columns={'ratings':'counts'}, inplace=True)
ratings['counts']
```




    0          5
    1          3
    2          3
    3          4
    4          5
              ..
    1000203    3
    1000205    5
    1000206    5
    1000207    4
    1000208    4
    Name: counts, Length: 836478, dtype: int64




```python
# 영화 제목을 보기 위해 메타 데이터를 읽어옵니다.
movie_file_path=os.getenv('HOME') + '/aiffel/recommendata_iu/data/ml-1m/movies.dat'
cols = ['movie_id', 'title', 'genre'] 
movies = pd.read_csv(movie_file_path, sep='::', names=cols, engine='python', encoding='ISO-8859-1')
movies.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>title</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children's|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



## 데이터 분석


```python
#ratings에 있는 유니크한 영화 개수
ratings['movie_id'].nunique()
```




    3628




```python
#ratings에 있는 유니크한 사용자 수
ratings['user_id'].nunique()
```




    6039




```python
movie_df = pd.merge(movies, ratings)
movie_df.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>title</th>
      <th>genre</th>
      <th>user_id</th>
      <th>counts</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
      <td>1</td>
      <td>5</td>
      <td>978824268</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
      <td>6</td>
      <td>4</td>
      <td>978237008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
      <td>8</td>
      <td>4</td>
      <td>978233496</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
      <td>9</td>
      <td>5</td>
      <td>978225952</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Animation|Children's|Comedy</td>
      <td>10</td>
      <td>5</td>
      <td>978226474</td>
    </tr>
  </tbody>
</table>
</div>




```python
#가장 인기있는 영화 30개(인기순)
movies_count = movie_df.groupby('title')['user_id'].count()
movies_count.sort_values(ascending=False).head(30)
```




    title
    American Beauty (1999)                                   3211
    Star Wars: Episode IV - A New Hope (1977)                2910
    Star Wars: Episode V - The Empire Strikes Back (1980)    2885
    Star Wars: Episode VI - Return of the Jedi (1983)        2716
    Saving Private Ryan (1998)                               2561
    Terminator 2: Judgment Day (1991)                        2509
    Silence of the Lambs, The (1991)                         2498
    Raiders of the Lost Ark (1981)                           2473
    Back to the Future (1985)                                2460
    Matrix, The (1999)                                       2434
    Jurassic Park (1993)                                     2413
    Sixth Sense, The (1999)                                  2385
    Fargo (1996)                                             2371
    Braveheart (1995)                                        2314
    Men in Black (1997)                                      2297
    Schindler's List (1993)                                  2257
    Princess Bride, The (1987)                               2252
    Shakespeare in Love (1998)                               2213
    L.A. Confidential (1997)                                 2210
    Shawshank Redemption, The (1994)                         2194
    Godfather, The (1972)                                    2167
    Groundhog Day (1993)                                     2121
    E.T. the Extra-Terrestrial (1982)                        2102
    Being John Malkovich (1999)                              2066
    Ghostbusters (1984)                                      2051
    Pulp Fiction (1994)                                      2030
    Forrest Gump (1994)                                      2022
    Terminator, The (1984)                                   2019
    Toy Story (1995)                                         2000
    Fugitive, The (1993)                                     1941
    Name: user_id, dtype: int64



## 선호하는 영화 추가하기


```python
my_favorite = ['Star Wars: Episode IV - A New Hope (1977)', 'Terminator 2: Judgment Day (1991)', 'Sixth Sense, The (1999)', 'Forrest Gump (1994)', 'Matrix, The (1999)']

my_list = pd.DataFrame({'user_id': ['braum']*5, 'title': my_favorite, 'counts':[5]*5})

if not movie_df.isin({'user_id':['braum']})['user_id'].any():  
    movie_df = movie_df.append(my_list)                          

movie_df.tail(10)  
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>title</th>
      <th>genre</th>
      <th>user_id</th>
      <th>counts</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>836473</th>
      <td>3952.0</td>
      <td>Contender, The (2000)</td>
      <td>Drama|Thriller</td>
      <td>5682</td>
      <td>3</td>
      <td>1.029458e+09</td>
    </tr>
    <tr>
      <th>836474</th>
      <td>3952.0</td>
      <td>Contender, The (2000)</td>
      <td>Drama|Thriller</td>
      <td>5812</td>
      <td>4</td>
      <td>9.920721e+08</td>
    </tr>
    <tr>
      <th>836475</th>
      <td>3952.0</td>
      <td>Contender, The (2000)</td>
      <td>Drama|Thriller</td>
      <td>5831</td>
      <td>3</td>
      <td>9.862231e+08</td>
    </tr>
    <tr>
      <th>836476</th>
      <td>3952.0</td>
      <td>Contender, The (2000)</td>
      <td>Drama|Thriller</td>
      <td>5837</td>
      <td>4</td>
      <td>1.011903e+09</td>
    </tr>
    <tr>
      <th>836477</th>
      <td>3952.0</td>
      <td>Contender, The (2000)</td>
      <td>Drama|Thriller</td>
      <td>5998</td>
      <td>4</td>
      <td>1.001781e+09</td>
    </tr>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Star Wars: Episode IV - A New Hope (1977)</td>
      <td>NaN</td>
      <td>braum</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Terminator 2: Judgment Day (1991)</td>
      <td>NaN</td>
      <td>braum</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Sixth Sense, The (1999)</td>
      <td>NaN</td>
      <td>braum</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Forrest Gump (1994)</td>
      <td>NaN</td>
      <td>braum</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Matrix, The (1999)</td>
      <td>NaN</td>
      <td>braum</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 사용하는 컬럼만 남기기
movie_df = movie_df[['user_id','title', 'counts']]
movie_df.sort_index()
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
      <th>user_id</th>
      <th>title</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>0</th>
      <td>braum</td>
      <td>Star Wars: Episode IV - A New Hope (1977)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>Toy Story (1995)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>braum</td>
      <td>Terminator 2: Judgment Day (1991)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>Toy Story (1995)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>836473</th>
      <td>5682</td>
      <td>Contender, The (2000)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>836474</th>
      <td>5812</td>
      <td>Contender, The (2000)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>836475</th>
      <td>5831</td>
      <td>Contender, The (2000)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>836476</th>
      <td>5837</td>
      <td>Contender, The (2000)</td>
      <td>4</td>
    </tr>
    <tr>
      <th>836477</th>
      <td>5998</td>
      <td>Contender, The (2000)</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>836483 rows × 3 columns</p>
</div>




```python
# 고유한 유저, 아티스트를 찾아내는 코드
user_unique = movie_df['user_id'].unique()
movie_unique = movie_df['title'].unique()

# 유저, 아티스트 indexing 하는 코드 
user_to_idx = {v:k for k,v in enumerate(user_unique)}
movie_to_idx = {v:k for k,v in enumerate(movie_unique)}
```


```python
print(user_to_idx['braum'])
print(movie_to_idx['Forrest Gump (1994)'])
```

    6039
    342



```python
temp_user_data = movie_df['user_id'].map(user_to_idx.get).dropna()
if len(temp_user_data) == len(movie_df):   # 모든 row가 정상적으로 인덱싱되었다면
    print('user_id column indexing OK!!')
    movie_df['user_id'] = temp_user_data   # data['user_id']을 인덱싱된 Series로 교체해 줍니다. 
else:
    print('user_id column indexing Fail!!')

# artist_to_idx을 통해 artist 컬럼도 동일한 방식으로 인덱싱해 줍니다. 
temp_movie_data = movie_df['title'].map(movie_to_idx.get).dropna()
if len(temp_movie_data) == len(movie_df):
    print('movie column indexing OK!!')
    movie_df['title'] = temp_movie_data
else:
    print('movie column indexing Fail!!')

movie_df
```

    user_id column indexing OK!!
    movie column indexing OK!!






<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>title</th>
      <th>counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>6039</td>
      <td>249</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6039</td>
      <td>569</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6039</td>
      <td>2507</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6039</td>
      <td>342</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6039</td>
      <td>2325</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>836483 rows × 3 columns</p>
</div>



## CSR Matrix 생성


```python
num_user = movie_df['user_id'].nunique()
num_movie = movie_df['title'].nunique()

csr_data = csr_matrix((movie_df.counts, (movie_df.user_id, movie_df.title)), shape=(num_user, num_movie)) 
csr_data
```




    <6040x3628 sparse matrix of type '<class 'numpy.int64'>'
    	with 836483 stored elements in Compressed Sparse Row format>



## als_model = AlternatingLeastSquares 모델을 직접 구성하여 훈련


```python
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['MKL_NUM_THREADS']='1'
```


```python
als_model = AlternatingLeastSquares(\
        factors=100, regularization=0.01, use_gpu=False, iterations=15, dtype=np.float32)
```


```python
csr_data_transpose = csr_data.T
csr_data_transpose
```




    <3628x6040 sparse matrix of type '<class 'numpy.int64'>'
    	with 836483 stored elements in Compressed Sparse Column format>




```python
als_model.fit(csr_data_transpose)
```


      0%|          | 0/15 [00:00<?, ?it/s]


## 선호도 파악


```python
braum, gump = user_to_idx['braum'], movie_to_idx['Forrest Gump (1994)']
braum_vector, gump_vector = als_model.user_factors[braum], als_model.item_factors[gump]
```


```python
braum_vector
```




    array([ 0.42408264, -0.13750367, -0.03471461,  0.1651544 , -0.47178945,
            0.61551124, -0.9676121 , -1.0181612 , -0.56242955,  0.48721218,
            0.14199786, -0.07294758,  0.5322658 , -0.07063968, -0.39939743,
           -0.13589108,  0.80609465,  0.88600886,  0.7028471 , -0.13961868,
            0.3711821 , -0.6554983 , -0.1855746 ,  0.14371157,  0.95859885,
           -0.23509519,  0.46512374, -0.04960321,  1.3953077 , -0.3573833 ,
           -0.73841256, -0.41173828,  0.09630907,  0.6274708 , -0.7867564 ,
           -0.08695952, -0.52107555,  0.01550765, -0.5685405 ,  0.10229736,
            0.16349936, -0.7629157 ,  0.23939282,  0.46038994,  1.0824912 ,
           -0.33896437,  1.0170999 ,  0.06521738,  0.66758966,  0.08094929,
            0.64356434, -0.4989629 , -0.5380442 ,  0.6306181 ,  0.74723595,
           -0.3189955 , -0.35359174, -0.3512256 ,  0.10463451,  0.45403466,
           -0.5122397 ,  0.45415482,  0.14621755,  0.18222666, -0.14287275,
            0.5465461 ,  0.4959907 , -1.0147961 , -0.34379435,  0.72856003,
            0.44144258,  0.29762527, -0.2773551 ,  0.6622691 ,  1.3065718 ,
           -0.5155282 ,  0.01828156,  0.49720994, -0.35006055, -1.1469206 ,
           -0.1749855 , -0.5897005 , -0.40239483, -0.81090736,  0.49007934,
           -0.34690356,  0.04317239,  1.0298615 , -0.41737032,  0.40720272,
           -0.6752725 ,  0.67274225, -0.4538166 , -0.59126896, -0.21973951,
            0.67505157, -0.66623634, -0.4280267 , -0.6942703 , -0.27093673],
          dtype=float32)




```python
gump_vector
```




    array([-0.00239462,  0.01376044, -0.01849912,  0.01024814,  0.03873129,
            0.02186281, -0.00624366, -0.0175504 , -0.02247315,  0.01202076,
            0.03007862, -0.01296514,  0.01905924,  0.00220958,  0.01233016,
           -0.00251718,  0.03877489,  0.02335927,  0.01976753, -0.03709752,
            0.01620624, -0.03408089,  0.01943785,  0.01486085,  0.03200487,
            0.01554504,  0.02441047,  0.02226409,  0.03458647, -0.00051717,
            0.01887115, -0.02701869,  0.02135779,  0.00907484, -0.00671072,
           -0.01197861,  0.04131725,  0.02519272, -0.02370717,  0.01128659,
            0.02032429, -0.01218636, -0.01141894,  0.03241245,  0.02759332,
           -0.02535917,  0.02818109, -0.01832988, -0.022521  ,  0.00115493,
            0.00467159,  0.01237522,  0.02281734, -0.0005395 ,  0.00869223,
           -0.01976266,  0.00287549, -0.00233529,  0.02549458,  0.00114005,
           -0.01100435, -0.00716755,  0.03929151,  0.02102371,  0.04626704,
            0.0260191 ,  0.01603343, -0.0037632 , -0.04320176,  0.01224567,
            0.02418424,  0.02473816, -0.01690205, -0.00567956,  0.03809491,
           -0.0005153 ,  0.02121704,  0.00168758, -0.02773681, -0.04707793,
            0.0042337 , -0.01211738,  0.02274987,  0.00567982,  0.00419963,
           -0.00079458,  0.02432861,  0.03888277,  0.00275434,  0.0124234 ,
            0.00691528,  0.01673841, -0.00976052, -0.01704242,  0.02488978,
           -0.00054543, -0.00284376, -0.00743458,  0.00515255,  0.00254498],
          dtype=float32)




```python
np.dot(braum_vector, gump_vector)
```




    0.5569296




```python
toy = movie_to_idx['Toy Story (1995)']
toy_vector = als_model.item_factors[toy]

print(np.dot(braum_vector, toy_vector))
```

    0.19235861


## 내가 좋아하는 영화와 비슷한 영화를 추천받기


```python
idx_to_movie = {v:k for k,v in movie_to_idx.items()}

def get_similar_movie(movie_name: str):
    movie_id = movie_to_idx[movie_name]
    similar_movie = als_model.similar_items(movie_id)
    similar_movie = [idx_to_movie[i[0]] for i in similar_movie]
    return similar_movie
```


```python
get_similar_movie('Forrest Gump (1994)') 
```




    ['Forrest Gump (1994)',
     'Groundhog Day (1993)',
     'Pretty Woman (1990)',
     'Sleepless in Seattle (1993)',
     'Four Weddings and a Funeral (1994)',
     'As Good As It Gets (1997)',
     'Clueless (1995)',
     'Pleasantville (1998)',
     'Wedding Singer, The (1998)',
     'Ghost (1990)']



## 내가 가장 좋아할 만한 영화들을 추천받기


```python
user = user_to_idx['braum']
movie_recommended = als_model.recommend(user, csr_data, N=20, filter_already_liked_items=True)

[idx_to_movie[i[0]] for i in movie_recommended]
```




    ['Star Wars: Episode VI - Return of the Jedi (1983)',
     'Star Wars: Episode V - The Empire Strikes Back (1980)',
     'Star Wars: Episode I - The Phantom Menace (1999)',
     'Jurassic Park (1993)',
     'Fugitive, The (1993)',
     'Terminator, The (1984)',
     'Men in Black (1997)',
     'Silence of the Lambs, The (1991)',
     'Groundhog Day (1993)',
     'Braveheart (1995)',
     'Back to the Future (1985)',
     'Total Recall (1990)',
     'American Beauty (1999)',
     'Hunt for Red October, The (1990)',
     'E.T. the Extra-Terrestrial (1982)',
     'L.A. Confidential (1997)',
     'Saving Private Ryan (1998)',
     'Alien (1979)',
     'Aliens (1986)',
     'Shakespeare in Love (1998)']




```python
gump = movie_to_idx['Forrest Gump (1994)']
explain = als_model.explain(user, csr_data, itemid=gump)
[(idx_to_movie[i[0]], i[1]) for i in explain[1]]
```




    [('Forrest Gump (1994)', 0.469374155461994),
     ('Sixth Sense, The (1999)', 0.07111166006796299),
     ('Star Wars: Episode IV - A New Hope (1977)', 0.02672230261740024),
     ('Terminator 2: Judgment Day (1991)', -0.0047077397708189615),
     ('Matrix, The (1999)', -0.01477341848380471)]


