# Transformer로 변역기 만들기

혁신적이었던 Seq2seq 구조로 번역기를 만들었으나 성능이 기대에 미치지 않았다.

Trasformer는 현재까지도 각종 번역 부문에서 최고의 성능을 자랑하는 모델이니 이번에는 정말 멋진 번역기를 만들 수 있을 것이다.

## 1. 모듈 import


```python
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import time
import re
import os
import io

from tqdm import tqdm
from tqdm import tqdm_notebook
import random

import sentencepiece as spm
from konlpy.tag import Mecab
```


```python
import matplotlib as mpl
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'
 
import matplotlib.font_manager as fm

fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()
```

## 2. 데이터 정제 및 토큰화


```python
data_dir = os.getenv('HOME') + '/aiffel/transformer/data/korean-english-park.train/'
kor_path = data_dir+"/korean-english-park.train.ko"
eng_path = data_dir+"/korean-english-park.train.en"

# 데이터 정제 및 토큰화
def clean_corpus(kor_path, eng_path):
    with open(kor_path, "r") as f: kor = f.read().splitlines()
    with open(eng_path, "r") as f: eng = f.read().splitlines()
    assert len(kor) == len(eng)  # kor, eng가 같은 갯수라는 것을 검증받기 위해 적용
    
    cleaned_corpus = list(set(zip(kor, eng)))  # 중복된 데이터 제거
    
    return cleaned_corpus

cleaned_corpus = clean_corpus(kor_path, eng_path)
```


```python
def preprocess_sentence(sentence):
    # 모든 입력을 소문자로 변환
    sentence = sentence.lower()
    # 알파벳, 문장부호, 한글만 남기고 모두 제거
    sentence = re.sub(r"[^a-zA-Z가-힣?.!,]+", " ", sentence)
    # 문장부호 양옆에 공백 추가
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    # 문장 앞뒤의 불필요한 공백 제거
    sentence = re.sub(r'[" "]+', " ", sentence)
    
    return sentence
```


```python
# Sentencepiece를 활용하여 학습한 tokenizer를 생성
def generate_tokenizer(corpus, vocab_size, lang="ko", pad_id=0, bos_id=1, eos_id=2, unk_id=3):
    # corpus를 받아 txt파일로 저장
    temp_file = os.getenv('HOME') + f'/aiffel/transformer/data/corpus_{lang}.txt'
    
    with open(temp_file, 'w') as f:
        for row in corpus:
            f.write(str(row) + '\n')
    
    # Sentencepiece를 이용해 
    spm.SentencePieceTrainer.Train(
        f'--input={temp_file} --pad_id={pad_id} --bos_id={bos_id} --eos_id={eos_id} \
        --unk_id={unk_id} --model_prefix=spm_{lang} --vocab_size={vocab_size}'
    )
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f'spm_{lang}.model')

    return tokenizer
```


```python
cleaned_corpus[0]
```




    ('젠 프사키 오바마측 선거운동 대변인은 젠 던햄이 9일 밤 자택에서 평화롭게 잠들었다고 전했다.',
     'Dunham passed away peacefully at her home shortly before midnight Sunday night (5 a.m. ET), campaign spokeswoman Jen Psaki told CNN.')




```python
SRC_VOCAB_SIZE = TGT_VOCAB_SIZE = 20000

eng_corpus = []
kor_corpus = []

for pair in cleaned_corpus:
    k, e = pair[0], pair[1]
    # kor, eng 나눠서 데이터 정제 후 분리
    kor_corpus.append(preprocess_sentence(k))
    eng_corpus.append(preprocess_sentence(e))

ko_tokenizer = generate_tokenizer(kor_corpus, SRC_VOCAB_SIZE, "ko")
en_tokenizer = generate_tokenizer(eng_corpus, TGT_VOCAB_SIZE, "en")
en_tokenizer.set_encode_extra_options("bos:eos")
```




    True




```python
src_corpus = []
tgt_corpus = []

assert len(kor_corpus) == len(eng_corpus)

# 토큰의 길이가 50 이하인 문장만 남김
for idx in tqdm(range(len(kor_corpus))):
    src = ko_tokenizer.EncodeAsIds(kor_corpus[idx])
    tgt = en_tokenizer.EncodeAsIds(eng_corpus[idx])
    
    if len(src) <= 50 and len(tgt) <= 50:
        src_corpus.append(src)
        tgt_corpus.append(tgt)

# 패딩처리를 완료하여 학습용 데이터를 완성
enc_train = tf.keras.preprocessing.sequence.pad_sequences(src_corpus, padding='post')
dec_train = tf.keras.preprocessing.sequence.pad_sequences(tgt_corpus, padding='post')
```

    100%|██████████| 78968/78968 [00:03<00:00, 21282.08it/s]


## 3. 모델 설계

### 3-1. Positional Encoding


```python
# pos - 단어가 위치한 Time-step(각각의 토큰의 위치정보값이며 정수값을 의미)
# d_model - 모델의 Embedding 차원 수
# i - Encoding차원의 index

def positional_encoding(pos, d_model):
    def cal_angle(position, i):
        return position / np.power(10000, int(i)/d_model)  # np.power(a,b) > a^b(제곱)
    
    def get_posi_angle_vec(position):
        return [cal_angle(position, i) for i in range(d_model)]
    
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(pos)])
    
    # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    
    return sinusoid_table
```

### 3-2. Multi-Head Attention


```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.depth = d_model // self.num_heads
        
        self.W_q = tf.keras.layers.Dense(d_model)  # Linear Layer
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        
        self.linear = tf.keras.layers.Dense(d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask):
        d_k = tf.cast(K.shape[-1], tf.float32)
        
        # Scaled QK 값 구하기
        QK = tf.matmul(Q, K, transpose_b=True)
        scaled_qk = QK / tf.math.sqrt(d_k)
        
        if mask is not None:
            scaled_qk += (mask * -1e9)
        
        # 1. Attention Weights 값 구하기 -> attentions
        attentions = tf.nn.softmax(scaled_qk, axis=-1)
        # 2. Attention 값을 V에 곱하기 -> out
        out = tf.matmul(attentions, V)
        return out, attentions
    
    def split_heads(self, x):
        """
        Embedding된 입력을 head의 수로 분할하는 함수
        
        x: [ batch x length x emb ]
        return: [ batch x length x heads x self.depth ]
        """
        bsz = x.shape[0]
        split_x = tf.reshape(x, (bsz, -1, self.num_heads, self.depth))
        split_x = tf.transpose(split_x, perm=[0, 2, 1, 3])
        return split_x
    
    def combine_heads(self, x):
        """
        분할된 Embedding을 하나로 결합하는 함수
        
        x: [ batch x length x heads x self.depth ]
        return: [ batch x length x emb ]
        """
        bsz = x.shape[0]
        combined_x = tf.transpose(x, perm=[0, 2, 1, 3])
        combined_x = tf.reshape(combined_x, (bsz, -1, self.d_model))
        return combined_x
    
    def call(self, Q, K, V, mask):
        """
        Step 1: Linear_in(Q, K, V) -> WQ, WK, WV
        Step 2: Split Heads(WQ, WK, WV) -> WQ_split, WK_split, WV_split
        Step 3: Scaled Dot Product Attention(WQ_split, WK_split, WV_split)
                 -> out, attention_weights
        Step 4: Combine Heads(out) -> out
        Step 5: Linear_out(out) -> out
        """
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

### 3-3. Position-wise Feed-Forward Network


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

### 3-4. Encoder Layer


```python
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.do = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, mask):
        # Multi-Head Attention
        residual = x
        out = self.norm_1(x)
        out, enc_attn = self.enc_self_attn(out, out, out, mask)
        out = self.do(out)
        out += residual
        
        # Position-Wise Feed Forward Network
        residual = out
        out = self.norm_2(out)
        out = self.ffn(out)
        out = self.do(out)
        out += residual
        
        return out, enc_attn
```

### 3-5. Decoder Layer


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

### 3-6. Encoder


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

### 3-7. Decoder


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

### 3-8. Transformer


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

### 3-9. Masking


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

## 4. 훈련하기


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
learning_rate = LearningRateScheduler(512)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
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
# Attention 시각화 함수
def visualize_attention(src, tgt, enc_attns, dec_attns, dec_enc_attns):
    def draw(data, ax, x="auto", y="auto"):
        import seaborn
        seaborn.heatmap(data, 
                        square=True,
                        vmin=0.0, vmax=1.0, 
                        cbar=False, ax=ax,
                        xticklabels=x,
                        yticklabels=y)
        
    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Encoder Layer", layer + 1)
        for h in range(4):
            draw(enc_attns[layer][0, h, :len(src), :len(src)], axs[h], src, src)
        plt.show()
        
    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Decoder Self Layer", layer+1)
        for h in range(4):
            draw(dec_attns[layer][0, h, :len(tgt), :len(tgt)], axs[h], tgt, tgt)
        plt.show()

        print("Decoder Src Layer", layer+1)
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        for h in range(4):
            draw(dec_enc_attns[layer][0, h, :len(tgt), :len(src)], axs[h], src, tgt)
        plt.show()
```


```python
# 번역 생성 함수
def evaluate(sentence, model, src_tokenizer, tgt_tokenizer):
    sentence = preprocess_sentence(sentence)
    pieces = src_tokenizer.encode_as_pieces(sentence)
    tokens = src_tokenizer.encode_as_ids(sentence)

    _input = tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen=enc_train.shape[-1], padding='post')

    ids = []
    output = tf.expand_dims([tgt_tokenizer.bos_id()], 0)
    for i in range(dec_train.shape[-1]):
        enc_padding_mask, combined_mask, dec_padding_mask = generate_masks(_input, output)
        
        # InvalidArgumentError: In[0] mismatch In[1] shape: 50 vs. 1: [1,8,1,50] [1,8,1,64] 0 0 [Op:BatchMatMulV2]
        predictions, enc_attns, dec_attns, dec_enc_attns = model(_input, output, enc_padding_mask, combined_mask, dec_padding_mask)
        
        predicted_id = tf.argmax(tf.math.softmax(predictions, axis=-1)[0, -1]).numpy().item()
        if tgt_tokenizer.eos_id() == predicted_id:
            result = tgt_tokenizer.decode_ids(ids)
            return pieces, result, enc_attns, dec_attns, dec_enc_attns

        ids.append(predicted_id)
        output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)
    result = tgt_tokenizer.decode_ids(ids)
    return pieces, result, enc_attns, dec_attns, dec_enc_attns
```


```python
# 번역 생성 및 Attention 시각화 결합
def translate(sentence, model, src_tokenizer, tgt_tokenizer, plot_attention=False):
    pieces, result, enc_attns, dec_attns, dec_enc_attns = evaluate(sentence, model, src_tokenizer, tgt_tokenizer)
    
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    if plot_attention:
        visualize_attention(pieces, result.split(), enc_attns, dec_attns, dec_enc_attns)
```


```python
examples = [
    "오바마는 대통령이다.",
    "시민들은 도시 속에 산다.",
    "커피는 필요 없다.",
    "일곱 명의 사망자가 발생했다."
]
```


```python
transformer = Transformer(
    n_layers=2,
    d_model=512,
    n_heads=8,
    d_ff=2048,
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    pos_len=200,
    dropout=0.2,
    shared=True
)
```


```python
# 학습
EPOCHS = 20
BATCH_SIZE = 32

for epoch in range(EPOCHS):
    total_loss = 0

    idx_list = list(range(0, enc_train.shape[0], BATCH_SIZE))
    random.shuffle(idx_list)
    t = tqdm(idx_list)

    for (batch, idx) in enumerate(t):
        batch_loss, enc_attns, dec_attns, dec_enc_attns = train_step(enc_train[idx:idx+BATCH_SIZE],
                                                                     dec_train[idx:idx+BATCH_SIZE],
                                                                     transformer,
                                                                     optimizer)

        total_loss += batch_loss

        t.set_description_str('Epoch %2d' % (epoch + 1))
        t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))
    
    # 매 Epoch 마다 제시된 예문에 대한 번역 생성
    for example in examples:
        translate(example, transformer, ko_tokenizer, en_tokenizer)
```

    Epoch  1: 100%|██████████| 2127/2127 [04:56<00:00,  7.16it/s, Loss 5.1270]


    Input: 오바마는 대통령이다.
    Predicted translation: obama has a presidential candidate in the presidential race .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: they are also also also between the two countries .
    Input: 커피는 필요 없다.
    Predicted translation: it was not clear .


    Epoch  2:   0%|          | 1/2127 [00:00<04:55,  7.19it/s, Loss 3.9358]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: the deaths of deaths were killed .


    Epoch  2: 100%|██████████| 2127/2127 [04:52<00:00,  7.28it/s, Loss 3.9443]


    Input: 오바마는 대통령이다.
    Predicted translation: it is the first time in the president .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city is in the city .
    Input: 커피는 필요 없다.
    Predicted translation: coffee coffee , coffees , coffee , coffee  coffee , coffee  coffee  coffees .


    Epoch  3:   0%|          | 1/2127 [00:00<05:13,  6.78it/s, Loss 3.2751]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: on thursday , the death toll has killed more than people .


    Epoch  3: 100%|██████████| 2127/2127 [04:52<00:00,  7.27it/s, Loss 3.4169]


    Input: 오바마는 대통령이다.
    Predicted translation: president elect barack obama .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city is the national mountain .
    Input: 커피는 필요 없다.
    Predicted translation: no need to be needed to be no needed to be needed to be no needed to be needed .


    Epoch  4:   0%|          | 1/2127 [00:00<05:05,  6.97it/s, Loss 1.9796]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: the death toll from the death toll in the latest incident .


    Epoch  4: 100%|██████████| 2127/2127 [04:52<00:00,  7.27it/s, Loss 2.8142]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is a great president .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city s city is getting the city .
    Input: 커피는 필요 없다.
    Predicted translation: even greener .


    Epoch  5:   0%|          | 1/2127 [00:00<04:53,  7.25it/s, Loss 2.2579]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: nine people were killed and injured .


    Epoch  5: 100%|██████████| 2127/2127 [04:52<00:00,  7.27it/s, Loss 2.2674]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city is in town .
    Input: 커피는 필요 없다.
    Predicted translation: coffee has no need for coffee .


    Epoch  6:   0%|          | 1/2127 [00:00<04:59,  7.11it/s, Loss 1.7310]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were dead and injured thursday .


    Epoch  6: 100%|██████████| 2127/2127 [04:52<00:00,  7.27it/s, Loss 1.7896]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is s advantage .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: they are in the city of lives in the city .
    Input: 커피는 필요 없다.
    Predicted translation: there is no need for coffee .


    Epoch  7:   0%|          | 1/2127 [00:00<05:01,  7.05it/s, Loss 1.1205]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were dead and injured .


    Epoch  7: 100%|██████████| 2127/2127 [04:52<00:00,  7.26it/s, Loss 1.3776]


    Input: 오바마는 대통령이다.
    Predicted translation: i think barack obama is a president .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city mayor is in city .
    Input: 커피는 필요 없다.
    Predicted translation: nothing need to worry about needs .


    Epoch  8:   0%|          | 1/2127 [00:00<05:04,  6.98it/s, Loss 0.8879]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were dead and seven others were injured last sunday .


    Epoch  8: 100%|██████████| 2127/2127 [04:52<00:00,  7.26it/s, Loss 1.0501]


    Input: 오바마는 대통령이다.
    Predicted translation: president barack obama is .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: city until there is loved city .
    Input: 커피는 필요 없다.
    Predicted translation: there s no need for coffee


    Epoch  9:   0%|          | 1/2127 [00:00<04:45,  7.44it/s, Loss 0.8751]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven were involved in the deadly clashes .


    Epoch  9: 100%|██████████| 2127/2127 [04:53<00:00,  7.26it/s, Loss 0.8154]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is the democratic nominee .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city is in city of city .
    Input: 커피는 필요 없다.
    Predicted translation: it s needed to show if you need .


    Epoch 10:   0%|          | 1/2127 [00:00<04:50,  7.32it/s, Loss 0.5552]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seventeen people were dead , and seven were older .


    Epoch 10: 100%|██████████| 2127/2127 [04:53<00:00,  7.26it/s, Loss 0.6605]


    Input: 오바마는 대통령이다.
    Predicted translation: president obama is .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city latest city visit the city
    Input: 커피는 필요 없다.
    Predicted translation: there is no need for coffee .


    Epoch 11:   0%|          | 1/2127 [00:00<05:04,  6.98it/s, Loss 0.2362]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven were killed sunday high .


    Epoch 11: 100%|██████████| 2127/2127 [04:53<00:00,  7.26it/s, Loss 0.5599]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is what is .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city late mark the city is love .
    Input: 커피는 필요 없다.
    Predicted translation: there need nothing needs to have to worry .


    Epoch 12:   0%|          | 1/2127 [00:00<04:54,  7.23it/s, Loss 0.1847]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven of the dead were wounded .


    Epoch 12: 100%|██████████| 2127/2127 [04:53<00:00,  7.25it/s, Loss 0.4912]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is what president is his soleway man .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city latestly in the city visit
    Input: 커피는 필요 없다.
    Predicted translation: nothing need is needed .


    Epoch 13:   0%|          | 1/2127 [00:00<04:59,  7.11it/s, Loss 0.4215]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: incidents occurred wednesday .


    Epoch 13: 100%|██████████| 2127/2127 [04:53<00:00,  7.24it/s, Loss 0.4401]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is aides .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city mayor is city of san francisco .
    Input: 커피는 필요 없다.
    Predicted translation: there s no coffee need for coffee .


    Epoch 14:   0%|          | 1/2127 [00:00<04:59,  7.11it/s, Loss 0.2103]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven were carried by senior officials wednesday .


    Epoch 14: 100%|██████████| 2127/2127 [04:53<00:00,  7.25it/s, Loss 0.4019]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city lately three urban drives in city .


      0%|          | 0/2127 [00:00<?, ?it/s]

    Input: 커피는 필요 없다.
    Predicted translation: there s no need for coffee .
    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were dead .


    Epoch 15: 100%|██████████| 2127/2127 [04:55<00:00,  7.19it/s, Loss 0.3731]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is what president .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city surpasss the city mayor in city .
    Input: 커피는 필요 없다.
    Predicted translation: there s no coffee need for coffee .


    Epoch 16:   0%|          | 1/2127 [00:00<04:47,  7.40it/s, Loss 0.2600]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were killed and seven others .


    Epoch 16: 100%|██████████| 2127/2127 [04:56<00:00,  7.18it/s, Loss 0.3472]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is what is a doctor .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: some city searching for the urban r .


      0%|          | 0/2127 [00:00<?, ?it/s]

    Input: 커피는 필요 없다.
    Predicted translation: there s nothing to drink coffee .
    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were dead .


    Epoch 17: 100%|██████████| 2127/2127 [04:56<00:00,  7.18it/s, Loss 0.3290]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is the democratic president .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city feels in the city is city .
    Input: 커피는 필요 없다.
    Predicted translation: there s a coffee need for coffee .


    Epoch 18:   0%|          | 1/2127 [00:00<05:01,  7.04it/s, Loss 0.0926]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were killed .


    Epoch 18: 100%|██████████| 2127/2127 [04:56<00:00,  7.18it/s, Loss 0.3105]


    Input: 오바마는 대통령이다.
    Predicted translation: president obama is .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: city officials in sanford .
    Input: 커피는 필요 없다.
    Predicted translation: it s needed .


    Epoch 19:   0%|          | 1/2127 [00:00<05:03,  7.01it/s, Loss 0.0939]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were killed and seven others .


    Epoch 19: 100%|██████████| 2127/2127 [04:55<00:00,  7.21it/s, Loss 0.3010]


    Input: 오바마는 대통령이다.
    Predicted translation: obama is the democratic nominee .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: the city captured the san residence of san francisco .
    Input: 커피는 필요 없다.
    Predicted translation: there s a coffee need for coffee .


    Epoch 20:   0%|          | 1/2127 [00:00<05:01,  7.05it/s, Loss 0.1223]

    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were confirmed dead .


    Epoch 20: 100%|██████████| 2127/2127 [04:53<00:00,  7.24it/s, Loss 0.2875]


    Input: 오바마는 대통령이다.
    Predicted translation: president barack obama is the last thing .
    Input: 시민들은 도시 속에 산다.
    Predicted translation: city last city . city i discover the city
    Input: 커피는 필요 없다.
    Predicted translation: there s a coffeer .
    Input: 일곱 명의 사망자가 발생했다.
    Predicted translation: seven people were killed by another person .


## 5. 결론

Loss 0.2875

Input: 오바마는 대통령이다.

Predicted translation: president barack obama is the last thing .

Input: 시민들은 도시 속에 산다.

Predicted translation: city last city . city i discover the city

Input: 커피는 필요 없다.

Predicted translation: there s a coffeer .

Input: 일곱 명의 사망자가 발생했다.

Predicted translation: seven people were killed by another person .

Seq2seq 구조를 이용한 번역기보다 성능이 향상되어 핵심 단어는 구현할 수 있었다.

# 회고
* 학습시간이 너무 오래 걸렸고, 많은 코드를 직접 구현 해야해서 어려웠던 프로젝트였다.
* 그래도 논문 구현을 했던 경험을 바탕으로 해서 이해하고 완성을 한 것에 대해 뿌듯함을 느낀다.
