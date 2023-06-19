# 챗봇 만들기

## 1. 모듈 import


```python
import os
import re
import random
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
from konlpy.tag import Mecab
from tqdm import tqdm
print(gensim.__version__)
```

    3.8.3



```python
csv_path = os.getenv('HOME') + '/aiffel/transformer_chatbot/Chatbot_data/ChatbotData.csv'
dataset = pd.read_csv(csv_path)
dataset.head()
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
      <th>Q</th>
      <th>A</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12시 땡!</td>
      <td>하루가 또 가네요.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1지망 학교 떨어졌어</td>
      <td>위로해 드립니다.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3박4일 놀러가고 싶다</td>
      <td>여행은 언제나 좋죠.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3박4일 정도 놀러가고 싶다</td>
      <td>여행은 언제나 좋죠.</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PPL 심하네</td>
      <td>눈살이 찌푸려지죠.</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 2. 데이터 정제

아래 조건을 만족하는 preprocess_sentence() 함수를 구현

영문자의 경우, 모두 소문자로 변환

영문자와 한글, 숫자, 그리고 주요 특수문자를 제외하곤 정규식을 활용하여 모두 제거


```python
dataset.drop_duplicates(inplace=True)
print(f"Data Num: {len(dataset):,}")
```

    Data Num: 11,823



```python
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z가-힣0-9?.!,]+", " ", sentence)
    return sentence
```


```python
dataset["Q"] = dataset["Q"].apply(preprocess_sentence)
dataset["A"] = dataset["A"].apply(preprocess_sentence)

display(dataset.sample(5))
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
      <th>Q</th>
      <th>A</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11500</th>
      <td>짝녀가 다른반인데 친해지는 방법없을까?</td>
      <td>다른 반 친구를 사겨보세요.</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11730</th>
      <td>커플여행 어떻게 생각해?</td>
      <td>누구랑 가느냐가 중요하겠죠.</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10046</th>
      <td>생각이 자꾸 나</td>
      <td>좋아하나봐요.</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10179</th>
      <td>썸 타는 거 티내고 싶진 않아.</td>
      <td>누구에게요?</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8035</th>
      <td>정말 모든게 그립지만.</td>
      <td>묻어두는 것도 좋겠지요.</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


## 3. 데이터 토큰화

토큰화에는 KoNLPy의 mecab 클래스를 사용

아래 조건을 만족하는 build_corpus() 함수를 구현

* 소스 문장 데이터와 타겟 문장 데이터를 입력으로 받는다.

* 데이터를 앞서 정의한 preprocess_sentence() 함수로 정제하고, 토큰화한다.

* 토큰화는 전달받은 토크나이즈 함수를 사용합니다. 이번엔 mecab.morphs 함수를 전달

* 토큰의 개수가 일정 길이 이상인 문장은 데이터에서 제외

* 중복되는 문장은 데이터에서 제외합니다. 소스 : 타겟 쌍을 비교하지 않고 소스는 소스대로 타겟은 타겟대로 검사합니다. 중복 쌍이 흐트러지지 않도록 유의


```python
m = Mecab()
```


```python
dataset["A"] = dataset["A"].apply(lambda x: "<sos> " + x + " <eos>")

display(dataset.sample(5))
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
      <th>Q</th>
      <th>A</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1946</th>
      <td>발목 접질렀어</td>
      <td>&lt;sos&gt; 꾸준히 치료하세요. &lt;eos&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11380</th>
      <td>짝남이 나한테 관심 없어보여도 계속 연락하고 들이대?</td>
      <td>&lt;sos&gt; 확실한 거절이 아니라면 부담스럽지 않은 선에서 연락하는게 좋겠어요. &lt;eos&gt;</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3372</th>
      <td>오랜만에 아침 먹었어</td>
      <td>&lt;sos&gt; 좋은 식습관이에요. &lt;eos&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2210</th>
      <td>비밀번호 자꾸 바꿔</td>
      <td>&lt;sos&gt; 보안상 그게 좋죠. &lt;eos&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10543</th>
      <td>여자들이 보통 좋아하는 음식이 뭐야?</td>
      <td>&lt;sos&gt; 사람 마다 다르겠지요. &lt;eos&gt;</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
def get_tokenizer(corpus, vocab_size):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='',
        oov_token="<UNK>",
        num_words=vocab_size
    )
    corpus_input = [sentence.split() for sentence in corpus]
    tokenizer.fit_on_texts(corpus_input)
    
    if vocab_size is not None:
        words_frequency = [w for w,c in tokenizer.word_index.items() if c >= vocab_size + 1]
        for w in words_frequency:
            del tokenizer.word_index[w]
            del tokenizer.word_counts[w]
    
    return tokenizer


concat = pd.concat([dataset["Q"], dataset["A"]])
tokenizer = get_tokenizer(concat, None)

print("Tokenizer Vocab Size:", f"{len(tokenizer.word_index):,}")
```

    Tokenizer Vocab Size: 21,745



```python
def encoding_sentence(copus, tokenizer):
    tensor = tokenizer.texts_to_sequences(copus)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(
        tensor, padding='post'
    )
    return tensor


enc_tensor = encoding_sentence(dataset["Q"], tokenizer)
dec_tensor = encoding_sentence(dataset["A"], tokenizer)
```


```python
def wordNumByFreq(tokenizer, freq_num):
    sorted_freq = sorted(tokenizer.word_counts.items(), key=lambda x: x[1])
    for idx, (_, freq) in enumerate(sorted_freq):
        if freq > freq_num: break;
    return idx
```

질문(Question) 문장의 길이가 15 이하이고 대답(Answer) 문장의 길이가 18 이하인 문장만 추출


```python
concat = pd.concat([dataset["Q"], dataset["A"]])
tokenizer = get_tokenizer(concat, 5872)

q = dataset["Q"].apply(lambda x: len(tokenizer.texts_to_sequences([x])[0]) <= 15)
a = dataset["A"].apply(lambda x: len(tokenizer.texts_to_sequences([x])[0]) <= 18)
dataset = dataset[q & a]

print("Tokenizer Vocab Size:", f"{len(tokenizer.word_index):,}")
```

    Tokenizer Vocab Size: 5,872



```python
test_dataset = dataset[:100]
dataset = dataset[100:]

display(test_dataset.sample(5))
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
      <th>Q</th>
      <th>A</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>sns 시간낭비인데 자꾸 보게됨</td>
      <td>&lt;sos&gt; 시간을 정하고 해보세요. &lt;eos&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3박4일 정도 놀러가고 싶다</td>
      <td>&lt;sos&gt; 여행은 언제나 좋죠. &lt;eos&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>개강룩 입어볼까</td>
      <td>&lt;sos&gt; 개시해보세요. &lt;eos&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>가족관계 알려 줘</td>
      <td>&lt;sos&gt; 저를 만들어 준 사람을 부모님, 저랑 이야기해 주는 사람을 친구로 생각하...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>가족이 누구야?</td>
      <td>&lt;sos&gt; 저를 만들어 준 사람을 부모님, 저랑 이야기해 주는 사람을 친구로 생각하...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


## 4. Augmentation

pip install --upgrade gensim==3.8.3

위의 실행문을 터미널에 입력하여 gensim의 버전을 다운그레이드 해줘서 ko.bin 파일 로드하기 


```python
word2vec_path = os.getenv('HOME') + '/aiffel/transformer_chatbot/Chatbot_data/ko.bin'
w2v = gensim.models.Word2Vec.load(word2vec_path)
```


```python
def lexical_sub(sentence, word2vec, enc_arg=True):
    toks = sentence.split()
    if not enc_arg:   #<sos>, <eos> 토큰 제외
        toks = toks[1:-1]

    _from = random.choice(toks)
    
    try:
        _to = word2vec.most_similar(_from)[0][0]
    except:
        return "_"
    
    res = ""
    for tok in sentence.split():
        if tok == _from:
            res += _to + " "
        else:
            res += tok + " "
    return res
```


```python
def argument_data(dataset, word2vec, enc_arg=True):
    qna = "Q" if enc_arg else "A"
    arg = dataset[qna].apply(lambda x: lexical_sub(x, word2vec, enc_arg))
    
    arg_data = dataset.copy()
    arg_data[qna] = arg
    
    arg_data = arg_data[arg_data[qna] != "_"]
    return arg_data
```


```python
enc_alpha = argument_data(dataset, w2v, True)
dec_alpha = argument_data(dataset, w2v, False)



enc_idx = set(dataset.index)
enc_alpha_idx = set(enc_alpha.index)
dec_alpha_idx = set(dec_alpha.index)

vet = enc_idx & enc_alpha_idx & dec_alpha_idx
vet = list(vet)[0]

print(f"Question Sentence: {dataset['Q'][vet]} ======> {enc_alpha['Q'][vet]}")
print(f"Answer Sentence: {dataset['A'][vet]} ======> {dec_alpha['A'][vet]}")
```

    /tmp/ipykernel_342/428797383.py:9: DeprecationWarning: Call to deprecated `most_similar` (Method will be removed in 4.0.0, use self.wv.most_similar() instead).
      _to = word2vec.most_similar(_from)[0][0]


    Question Sentence: 벚꽃이 너무 예뻐 ======> 벚꽃이 워낙 예뻐 
    Answer Sentence: <sos> 너무 아름답죠. <eos> ======> <sos> 워낙 아름답죠. <eos> 



```python
dataset = pd.concat([dataset, enc_alpha, dec_alpha])
dataset = dataset.sample(frac=1)

print(f"Dataset Num: {len(dataset):,}")
display(dataset[:5])
```

    Dataset Num: 17,423



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
      <th>Q</th>
      <th>A</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4296</th>
      <td>지하철에 사람이 너무 많아</td>
      <td>&lt;sos&gt; 맨 앞이나 맨 뒤에 타세요. &lt;eos&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5310</th>
      <td>1년이 아무것도 아니였나</td>
      <td>&lt;sos&gt; 아무것도 아닌 건 아닐 거예요. &lt;eos&gt;</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3553</th>
      <td>운명인가</td>
      <td>&lt;sos&gt; 인연인가 봐요. &lt;eos&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10195</th>
      <td>썸 타는 사람 있는데 망설여지는데</td>
      <td>&lt;sos&gt; 어떤 부분이 망설여지는지 말씀해보세요. &lt;eos&gt;</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11528</th>
      <td>짝녀랑 드디어 사귄다!</td>
      <td>&lt;sos&gt; 좋은 소식이네요. &lt;eos&gt;</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
enc_tensor = encoding_sentence(dataset["Q"], tokenizer)
dec_tensor = encoding_sentence(dataset["A"], tokenizer)

print("Data num:", f"{len(enc_tensor):,}")
```

    Data num: 17,423


## 5. 훈련하기

### Positional Encoding


```python
def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, int(i)/d_model)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]
    
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])
    
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    
    return sinusoid_table
```

### Multi-Head Attention


```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
            
        self.depth = d_model // self.num_heads
            
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
            
        self.linear = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask):
        d_k = tf.cast(K.shape[-1], tf.float32)
        QK = tf.matmul(Q, K, transpose_b=True)

        scaled_qk = QK / tf.math.sqrt(d_k)

        if mask is not None: scaled_qk += (mask * -1e9)  

        attentions = tf.nn.softmax(scaled_qk, axis=-1)
        out = tf.matmul(attentions, V)

        return out, attentions
            

    def split_heads(self, x):
        batch_size = x.shape[0]
        split_x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        split_x = tf.transpose(split_x, perm=[0, 2, 1, 3])

        return split_x

    def combine_heads(self, x):
        batch_size = x.shape[0]
        combined_x = tf.transpose(x, perm=[0, 2, 1, 3])
        combined_x = tf.reshape(combined_x, (batch_size, -1, self.d_model))

        return combined_x

        
    def call(self, Q, K, V, mask):
        WQ = self.W_q(Q)
        WK = self.W_k(K)
        WV = self.W_v(V)
        
        WQ_splits = self.split_heads(WQ)
        WK_splits = self.split_heads(WK)
        WV_splits = self.split_heads(WV)
            
        out, attention_weights = self.scaled_dot_product_attention(
            WQ_splits, WK_splits, WV_splits, mask
        )
        out = self.combine_heads(out)
        out = self.linear(out)
                
        return out, attention_weights
```

### Position-wise Feed-Forward Network


```python
class PoswiseFeedForwardNet(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.w_2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        out = self.w_1(x)
        out = self.w_2(out)
            
        return out
```

### Encoder Layer


```python
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, mask):
        residual = x
        out = self.norm_1(x)
        out, enc_attn = self.enc_self_attn(out, out, out, mask)
        out = self.dropout(out)
        out += residual
        
        residual = out
        out = self.norm_2(out)
        out = self.ffn(out)
        out = self.dropout(out)
        out += residual
        
        return out, enc_attn
```

### Decoder Layer


```python
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.do = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, enc_out, causality_mask, padding_mask):
        # Masked Multi-Head Attention
        residual = x
        out = self.norm_1(x)
        #out, dec_attn = self.dec_self_attn(out, out, out, causality_mask)
        out, dec_attn = self.dec_self_attn(out, out, out, padding_mask)
        out = self.do(out)
        out += residual
        
        # Multi-Head Attention
        residual = out
        out = self.norm_2(out)
        #out, dec_enc_attn = self.enc_dec_attn(out, enc_out, enc_out, padding_mask)
        out, dec_enc_attn = self.enc_dec_attn(out, enc_out, enc_out, causality_mask)
        out = self.do(out)
        out += residual

        # Position-Wise Feed Forward Network
        residual = out
        out = self.norm_3(out)
        out = self.ffn(out)
        out = self.do(out)
        out += residual

        return out, dec_attn, dec_enc_attn
```

### Encoder


```python
class Encoder(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.enc_layers = [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.do = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, mask):
        out = x
        enc_attns = list()
        for i in range(self.n_layers):
            out, enc_attn = self.enc_layers[i](out, mask)
            enc_attns.append(enc_attn)
            
        return out, enc_attns
```

### Decoder


```python
class Decoder(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.dec_layers = [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        
    def call(self, x, enc_out, causality_mask, padding_mask):
        out = x
        dec_attns = list()
        dec_enc_attns = list()
        for i in range(self.n_layers):
            out, dec_attn, dec_enc_attn = self.dec_layers[i](out, enc_out, causality_mask, padding_mask)
            dec_attns.append(dec_attn)
            dec_enc_attns.append(dec_enc_attn)
        
        return out, dec_attns, dec_enc_attns
```

### Transformer


```python
class Transformer(tf.keras.Model):
    def __init__(self, n_layers, d_model, n_heads, d_ff, src_vocab_size, tgt_vocab_size,
                 pos_len, dropout=0.2, shared=True):
        super(Transformer, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        
        # 1. Embedding Layer 정의
        self.enc_emb = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.dec_emb = tf.keras.layers.Embedding(tgt_vocab_size, d_model)
        
        # 2. Positional Encoding 정의
        self.pos_encoding = positional_encoding(pos_len, d_model)
        # 6. Dropout 정의
        self.do = tf.keras.layers.Dropout(dropout)
        
        # 3. Encoder / Decoder 정의
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        
        # 4. Output Linear 정의
        self.fc = tf.keras.layers.Dense(tgt_vocab_size)
        
        # 5. Shared Weights
        self.shared = shared
        
        if shared:
            self.fc.set_weights(tf.transpose(self.dec_emb.weights))
        
        
    def embedding(self, emb, x):
        """
        입력된 정수 배열을 Embedding + Pos Encoding
        + Shared일 경우 Scaling 작업 포함

        x: [ batch x length ]
        return: [ batch x length x emb ]
        """
        seq_len = x.shape[1]
        out = emb(x)
        
        if self.shared:
            out *= tf.math.sqrt(self.d_model)
        
        out += self.pos_encoding[np.newaxis, ...][:, :seq_len, :]
        out = self.do(out)
        
        return out
    
    def call(self, enc_in, dec_in, enc_mask, causality_mask, dec_mask):
        # Step 1: Embedding(enc_in, dec_in) -> enc_in, dec_in
        enc_in = self.embedding(self.enc_emb, enc_in)
        dec_in = self.embedding(self.dec_emb, dec_in)
        # Step 2: Encoder(enc_in, enc_mask) -> enc_out, enc_attns
        enc_out, enc_attns = self.encoder(enc_in, enc_mask)
        # Step 3: Decoder(dec_in, enc_out, mask) -> dec_out, dec_attns, dec_enc_attns
        dec_out, dec_attns, dec_enc_attns = self.decoder(dec_in, enc_out, causality_mask, dec_mask)
        # Step 4: Out Linear(dec_out) -> logits
        logits = self.fc(dec_out)
        return logits, enc_attns, dec_attns, dec_enc_attns
```

### Masking


```python
# Attention을 할 때에 <PAD> 토큰에도 Attention을 주는 것을 방지해 주는 역할
def generate_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def generate_causality_mask(src_len, tgt_len):
    mask = 1 - np.cumsum(np.eye(src_len, tgt_len), 0)
    return tf.cast(mask, tf.float32)

def generate_masks(src, tgt):
    enc_mask = generate_padding_mask(src)
    dec_mask = generate_padding_mask(tgt)

    dec_enc_causality_mask = generate_causality_mask(tgt.shape[1], src.shape[1])
    dec_enc_mask = tf.maximum(enc_mask, dec_enc_causality_mask)

    dec_causality_mask = generate_causality_mask(tgt.shape[1], tgt.shape[1])
    dec_mask = tf.maximum(dec_mask, dec_causality_mask)

    return enc_mask, dec_enc_mask, dec_mask
```


```python
class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateScheduler, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)
```


```python
# Loss 함수 정의
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    # Masking 되지 않은 입력의 개수로 Scaling하는 과정
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
```


```python
learning_rate = LearningRateScheduler(512)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
```


```python
@tf.function()
def train_step(src, tgt, model, optimizer):
    gold = tgt[:, 1:]
        
    enc_mask, dec_enc_mask, dec_mask = generate_masks(src, tgt)

    # 계산된 loss에 tf.GradientTape()를 적용해 학습을 진행합니다.
    with tf.GradientTape() as tape:
        predictions, enc_attns, dec_attns, dec_enc_attns = model(src, tgt, enc_mask, dec_enc_mask, dec_mask)
        loss = loss_function(gold, predictions[:, :-1])

    # 최종적으로 optimizer.apply_gradients()가 사용됩니다. 
    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, enc_attns, dec_attns, dec_enc_attns
```


```python
transformer = Transformer(
    n_layers=2,
    d_model=128,
    n_heads=8,
    d_ff=256,
    src_vocab_size=5872,
    tgt_vocab_size=5872,
    pos_len=200,
    dropout=0.5,
    shared=True
)
```


```python
def model_fit(enc_train, dec_train, model, epochs, batch_size):
    for epoch in range(epochs):
        total_loss = 0

        idx_list = list(range(0, enc_train.shape[0], batch_size))
        random.shuffle(idx_list)
        t = tqdm(idx_list)

        for (batch, idx) in enumerate(t):
            batch_loss, enc_attns, dec_attns, dec_enc_attns = \
            train_step(
                enc_train[idx:idx+batch_size],
                dec_train[idx:idx+batch_size],
                model,
                optimizer
            )

            total_loss += batch_loss

            t.set_description_str('Epoch %2d' % (epoch + 1))
            t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))
            

model_fit(enc_tensor, dec_tensor, transformer, epochs=10, batch_size=64)
```

    Epoch  1: 100%|██████████| 273/273 [00:10<00:00, 26.31it/s, Loss 6.9641]
    Epoch  2: 100%|██████████| 273/273 [00:03<00:00, 73.41it/s, Loss 5.2730]
    Epoch  3: 100%|██████████| 273/273 [00:03<00:00, 72.29it/s, Loss 4.8571]
    Epoch  4: 100%|██████████| 273/273 [00:03<00:00, 73.17it/s, Loss 4.4409]
    Epoch  5: 100%|██████████| 273/273 [00:03<00:00, 73.17it/s, Loss 3.9131]
    Epoch  6: 100%|██████████| 273/273 [00:03<00:00, 71.02it/s, Loss 3.2623]
    Epoch  7: 100%|██████████| 273/273 [00:03<00:00, 72.57it/s, Loss 2.6414]
    Epoch  8: 100%|██████████| 273/273 [00:03<00:00, 72.87it/s, Loss 2.1339]
    Epoch  9: 100%|██████████| 273/273 [00:03<00:00, 71.11it/s, Loss 1.7424]
    Epoch 10: 100%|██████████| 273/273 [00:03<00:00, 72.06it/s, Loss 1.4559]


## 6. 성능 측정하기


```python
def translate(sentence, model, tokenizer, enc_tensor, dec_tensor):
    enc_maxlen = enc_tensor.shape[-1]
    dec_maxlen = dec_tensor.shape[-1]

    sos_idx = tokenizer.word_index['<sos>']
    eos_idx = tokenizer.word_index['<eos>']

    sentence = preprocess_sentence(sentence)

    m = Mecab()
    sentence = m.morphs(sentence)

    _input = tokenizer.texts_to_sequences([sentence])
    _input = tf.keras.preprocessing.sequence.pad_sequences(
        _input,
        maxlen=enc_maxlen,
        padding='post'
    )

    ids = []
    output = tf.expand_dims([sos_idx], 0)

    for i in range(dec_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = generate_masks(
            _input, output
        )

        predictions, enc_attns, dec_attns, dec_enc_attns = model(
            _input, output, enc_padding_mask, combined_mask, dec_padding_mask
        )

        predicted_id = tf.argmax(
            tf.math.softmax(predictions, axis=-1)[0, -1]
        ).numpy().item()

        if predicted_id == eos_idx:
            result = tokenizer.sequences_to_texts([ids])
            return result

        ids.append(predicted_id)
        output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)
    result = tokenizer.sequences_to_texts([ids])
    return result


print("=" * 100)
test_sentences = [
    "지루하다, 놀러가고 싶어.",
    "오늘 일찍 일어났더니 피곤하다.",
    "간만에 여자친구랑 데이트 하기로 했어.",
    "집에 있는다는 소리야."
]

for sentence in test_sentences:
    ans = translate(sentence, transformer, tokenizer, enc_tensor, dec_tensor)[0]
    print(f"Quenstion: {sentence:<30}\tAnswer: {ans:<30}")
print("=" * 100)
```

    ====================================================================================================
    Quenstion: 지루하다, 놀러가고 싶어.                	Answer: <UNK> 많이 <UNK>                
    Quenstion: 오늘 일찍 일어났더니 피곤하다.             	Answer: <UNK> 시간이 <UNK>               
    Quenstion: 간만에 여자친구랑 데이트 하기로 했어.         	Answer: <UNK> <UNK>                   
    Quenstion: 집에 있는다는 소리야.                  	Answer: <UNK> 싶지 않아요.                 
    ====================================================================================================


## 7. 결론

굉장히 어려운 노드였다.

선배님들의 코드를 참고하면서 했는데도 결과가 잘 나오지 않았다.
