# 번역기 만들기

## 데이터 가져오기


```python
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import numpy as np
import os
import re
```


```python
file_path = os.getenv('HOME')+'/aiffel/translator_seq2seq/data/fra.txt'
lines = pd.read_csv(file_path, names=['eng', 'fra', 'cc'], sep='\t')
print('전체 샘플의 수 :',len(lines))
lines.sample(5) #샘플 5개 출력
```

    전체 샘플의 수 : 217975





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
      <th>eng</th>
      <th>fra</th>
      <th>cc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29994</th>
      <td>Whose cup is this?</td>
      <td>C’est à qui, cette tasse ?</td>
      <td>CC-BY 2.0 (France) Attribution: tatoeba.org #8...</td>
    </tr>
    <tr>
      <th>84938</th>
      <td>You won't bleed to death.</td>
      <td>Tu ne saigneras pas jusqu'à ce que mort s'ensu...</td>
      <td>CC-BY 2.0 (France) Attribution: tatoeba.org #2...</td>
    </tr>
    <tr>
      <th>178246</th>
      <td>You don't need to think about that now.</td>
      <td>Vous n'avez pas besoin d'y penser maintenant.</td>
      <td>CC-BY 2.0 (France) Attribution: tatoeba.org #7...</td>
    </tr>
    <tr>
      <th>1212</th>
      <td>I'm lucky.</td>
      <td>Je suis veinarde.</td>
      <td>CC-BY 2.0 (France) Attribution: tatoeba.org #2...</td>
    </tr>
    <tr>
      <th>126550</th>
      <td>Why are they doing this to me?</td>
      <td>Pourquoi me font-ils ça ?</td>
      <td>CC-BY 2.0 (France) Attribution: tatoeba.org #5...</td>
    </tr>
  </tbody>
</table>
</div>




```python
lines = lines[['eng', 'fra']][:33000] # 5만개 샘플 사용
lines.sample(5)
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
      <th>eng</th>
      <th>fra</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8050</th>
      <td>I'm a dentist.</td>
      <td>Je suis dentiste.</td>
    </tr>
    <tr>
      <th>31320</th>
      <td>Everybody did that.</td>
      <td>Tout le monde a fait ça.</td>
    </tr>
    <tr>
      <th>28496</th>
      <td>They were panting.</td>
      <td>Elles tiraient la langue.</td>
    </tr>
    <tr>
      <th>27338</th>
      <td>It's unauthorized.</td>
      <td>Ce n'est pas autorisé.</td>
    </tr>
    <tr>
      <th>14819</th>
      <td>I caught a cold.</td>
      <td>J'ai contracté un rhume.</td>
    </tr>
  </tbody>
</table>
</div>



## 데이터 전처리


```python
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
  
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!]+", " ", sentence)
    
    sentence = sentence.strip()
    sentence = sentence.split(" ")
    
    return sentence
```


```python
def preprocess_sentence_decoder(sentence):
    sentence = sentence.lower().strip()
  
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!]+", " ", sentence)
    
    sentence = sentence.strip()
    sentence = '<start> ' + sentence + ' <end>'
    sentence = sentence.split(" ")
    
    return sentence
```


```python
lines.eng = lines.eng.apply(lambda x : preprocess_sentence(x))
lines.fra = lines.fra.apply(lambda x : preprocess_sentence_decoder(x))
```


```python
lines.eng.sample(5)
```




    22789        [tom, knows, better, .]
    18752       [can, you, prove, it, ?]
    32460    [i, have, to, go, there, .]
    23966       [you, may, go, there, .]
    855                   [we, agree, .]
    Name: eng, dtype: object




```python
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(lines.eng)
```


```python
input_text = eng_tokenizer.texts_to_sequences(lines.eng)
input_text[:3]
```




    [[27, 1], [27, 1], [27, 1]]




```python
fra_tokenizer = Tokenizer()
fra_tokenizer.fit_on_texts(lines.fra)
target_text = fra_tokenizer.texts_to_sequences(lines.fra)
```


```python
target_text[:3]
```




    [[1, 72, 9, 2], [1, 340, 3, 2], [1, 27, 523, 9, 2]]




```python
eng_vocab_size = len(eng_tokenizer.word_index) + 1
fra_vocab_size = len(fra_tokenizer.word_index) + 1
print('영어 단어장의 크기 :', eng_vocab_size)
print('프랑스어 단어장의 크기 :', fra_vocab_size)
```

    영어 단어장의 크기 : 4514
    프랑스어 단어장의 크기 : 7263



```python
max_eng_seq_len = max([len(line) for line in input_text])
max_fra_seq_len = max([len(line) for line in target_text])
print('영어 시퀀스의 최대 길이', max_eng_seq_len)
print('프랑스어 시퀀스의 최대 길이', max_fra_seq_len)
```

    영어 시퀀스의 최대 길이 8
    프랑스어 시퀀스의 최대 길이 17



```python
print('전체 샘플의 수 :',len(lines))
print('영어 단어장의 크기 :', eng_vocab_size)
print('프랑스어 단어장의 크기 :', fra_vocab_size)
print('영어 시퀀스의 최대 길이', max_eng_seq_len)
print('프랑스어 시퀀스의 최대 길이', max_fra_seq_len)
```

    전체 샘플의 수 : 33000
    영어 단어장의 크기 : 4514
    프랑스어 단어장의 크기 : 7263
    영어 시퀀스의 최대 길이 8
    프랑스어 시퀀스의 최대 길이 17



```python
sos_token = '<start>'
eos_token = '<end>'

encoder_input = input_text
# 종료 토큰 제거
decoder_input = [[ char for char in line if char != fra_tokenizer.word_index[eos_token] ] for line in target_text] 
# 시작 토큰 제거
decoder_target = [[ char for char in line if char != fra_tokenizer.word_index[sos_token] ] for line in target_text]
```


```python
encoder_input = pad_sequences(encoder_input, maxlen = max_eng_seq_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen = max_fra_seq_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen = max_fra_seq_len, padding='post')
print('영어 데이터의 크기(shape) :',np.shape(encoder_input))
print('프랑스어 입력데이터의 크기(shape) :',np.shape(decoder_input))
print('프랑스어 출력데이터의 크기(shape) :',np.shape(decoder_target))
```

    영어 데이터의 크기(shape) : (33000, 8)
    프랑스어 입력데이터의 크기(shape) : (33000, 17)
    프랑스어 출력데이터의 크기(shape) : (33000, 17)



```python
indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)

encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]
```


```python
n_of_val = 3000

encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]

print(encoder_input_train.shape)
print(decoder_input_train.shape)
print(decoder_target_train.shape)
print(encoder_input_test.shape)
print(decoder_input_test.shape)
print(decoder_target_test.shape)
```

    (30000, 8)
    (30000, 17)
    (30000, 17)
    (3000, 8)
    (3000, 17)
    (3000, 17)


## 모델 훈련


```python
encoder_inputs = Input(shape=(None,))
# encoder embedding
enc_emb = Embedding(eng_vocab_size, 256, input_length=max_eng_seq_len)(encoder_inputs)
enc_masking = Masking(mask_value=0.0)(enc_emb)
encoder_lstm = LSTM(units = 256, return_state = True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_masking)
encoder_states = [state_h, state_c]
```


```python
decoder_inputs = Input(shape=(None,))
# decoder embedding
dec_emb = Embedding(fra_vocab_size, 256)(decoder_inputs)
dec_masking = Masking(mask_value=0.0)(dec_emb)
decoder_lstm = LSTM(units = 256, return_sequences = True, return_state=True)
decoder_outputs, _, _= decoder_lstm(dec_masking, initial_state = encoder_states)

decoder_softmax_layer = Dense(fra_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)
```


```python
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    input_2 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    embedding (Embedding)           (None, None, 256)    1155584     input_1[0][0]                    
    __________________________________________________________________________________________________
    embedding_1 (Embedding)         (None, None, 256)    1859328     input_2[0][0]                    
    __________________________________________________________________________________________________
    masking (Masking)               (None, None, 256)    0           embedding[0][0]                  
    __________________________________________________________________________________________________
    masking_1 (Masking)             (None, None, 256)    0           embedding_1[0][0]                
    __________________________________________________________________________________________________
    lstm (LSTM)                     [(None, 256), (None, 525312      masking[0][0]                    
    __________________________________________________________________________________________________
    lstm_1 (LSTM)                   [(None, None, 256),  525312      masking_1[0][0]                  
                                                                     lstm[0][1]                       
                                                                     lstm[0][2]                       
    __________________________________________________________________________________________________
    dense (Dense)                   (None, None, 7263)   1866591     lstm_1[0][0]                     
    ==================================================================================================
    Total params: 5,932,127
    Trainable params: 5,932,127
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy', metrics=['acc'])
```


```python
model.fit([encoder_input_train, decoder_input_train],decoder_target_train,
          validation_data = ([encoder_input_test, decoder_input_test], decoder_target_test),
          batch_size = 32, epochs = 50)
```

    Epoch 1/50
    938/938 [==============================] - 50s 20ms/step - loss: 1.4439 - acc: 0.7739 - val_loss: 1.1834 - val_acc: 0.8138
    Epoch 2/50
    938/938 [==============================] - 16s 18ms/step - loss: 1.0895 - acc: 0.8260 - val_loss: 1.0375 - val_acc: 0.8346
    Epoch 3/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.9652 - acc: 0.8435 - val_loss: 0.9511 - val_acc: 0.8475
    Epoch 4/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.8815 - acc: 0.8547 - val_loss: 0.8872 - val_acc: 0.8562
    Epoch 5/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.8203 - acc: 0.8639 - val_loss: 0.8447 - val_acc: 0.8629
    Epoch 6/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.7680 - acc: 0.8720 - val_loss: 0.8154 - val_acc: 0.8663
    Epoch 7/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.7253 - acc: 0.8789 - val_loss: 0.7927 - val_acc: 0.8717
    Epoch 8/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.6948 - acc: 0.8850 - val_loss: 0.7844 - val_acc: 0.8744
    Epoch 9/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.6730 - acc: 0.8904 - val_loss: 0.7836 - val_acc: 0.8749
    Epoch 10/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.6529 - acc: 0.8957 - val_loss: 0.7733 - val_acc: 0.8788
    Epoch 11/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.6396 - acc: 0.8999 - val_loss: 0.7714 - val_acc: 0.8808
    Epoch 12/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.6229 - acc: 0.9038 - val_loss: 0.7604 - val_acc: 0.8819
    Epoch 13/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.6041 - acc: 0.9073 - val_loss: 0.7657 - val_acc: 0.8823
    Epoch 14/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.5972 - acc: 0.9104 - val_loss: 0.7674 - val_acc: 0.8837
    Epoch 15/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.5903 - acc: 0.9128 - val_loss: 0.7707 - val_acc: 0.8831
    Epoch 16/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.5802 - acc: 0.9147 - val_loss: 0.7607 - val_acc: 0.8837
    Epoch 17/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.5655 - acc: 0.9173 - val_loss: 0.7624 - val_acc: 0.8843
    Epoch 18/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.5622 - acc: 0.9187 - val_loss: 0.7680 - val_acc: 0.8839
    Epoch 19/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.5582 - acc: 0.9202 - val_loss: 0.7678 - val_acc: 0.8840
    Epoch 20/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.5526 - acc: 0.9217 - val_loss: 0.7700 - val_acc: 0.8843
    Epoch 21/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.5453 - acc: 0.9231 - val_loss: 0.7673 - val_acc: 0.8847
    Epoch 22/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.5376 - acc: 0.9243 - val_loss: 0.7672 - val_acc: 0.8842
    Epoch 23/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.5294 - acc: 0.9253 - val_loss: 0.7650 - val_acc: 0.8842
    Epoch 24/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.5232 - acc: 0.9262 - val_loss: 0.7652 - val_acc: 0.8845
    Epoch 25/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.5159 - acc: 0.9273 - val_loss: 0.7594 - val_acc: 0.8856
    Epoch 26/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.5093 - acc: 0.9283 - val_loss: 0.7588 - val_acc: 0.8857
    Epoch 27/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.5028 - acc: 0.9292 - val_loss: 0.7592 - val_acc: 0.8856
    Epoch 28/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4970 - acc: 0.9297 - val_loss: 0.7572 - val_acc: 0.8846
    Epoch 29/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4915 - acc: 0.9304 - val_loss: 0.7562 - val_acc: 0.8850
    Epoch 30/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.4857 - acc: 0.9309 - val_loss: 0.7581 - val_acc: 0.8851
    Epoch 31/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.4805 - acc: 0.9319 - val_loss: 0.7531 - val_acc: 0.8855
    Epoch 32/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4745 - acc: 0.9324 - val_loss: 0.7518 - val_acc: 0.8859
    Epoch 33/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4701 - acc: 0.9325 - val_loss: 0.7496 - val_acc: 0.8859
    Epoch 34/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4651 - acc: 0.9334 - val_loss: 0.7521 - val_acc: 0.8862
    Epoch 35/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4610 - acc: 0.9340 - val_loss: 0.7523 - val_acc: 0.8858
    Epoch 36/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.4565 - acc: 0.9342 - val_loss: 0.7513 - val_acc: 0.8852
    Epoch 37/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4531 - acc: 0.9345 - val_loss: 0.7529 - val_acc: 0.8847
    Epoch 38/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.4487 - acc: 0.9351 - val_loss: 0.7518 - val_acc: 0.8855
    Epoch 39/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.4457 - acc: 0.9355 - val_loss: 0.7524 - val_acc: 0.8858
    Epoch 40/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.4425 - acc: 0.9357 - val_loss: 0.7560 - val_acc: 0.8850
    Epoch 41/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.4393 - acc: 0.9362 - val_loss: 0.7570 - val_acc: 0.8841
    Epoch 42/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4370 - acc: 0.9362 - val_loss: 0.7599 - val_acc: 0.8848
    Epoch 43/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4340 - acc: 0.9366 - val_loss: 0.7566 - val_acc: 0.8851
    Epoch 44/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4308 - acc: 0.9369 - val_loss: 0.7600 - val_acc: 0.8851
    Epoch 45/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.4282 - acc: 0.9373 - val_loss: 0.7572 - val_acc: 0.8854
    Epoch 46/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4252 - acc: 0.9376 - val_loss: 0.7602 - val_acc: 0.8848
    Epoch 47/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.4227 - acc: 0.9376 - val_loss: 0.7623 - val_acc: 0.8850
    Epoch 48/50
    938/938 [==============================] - 16s 18ms/step - loss: 0.4201 - acc: 0.9381 - val_loss: 0.7578 - val_acc: 0.8856
    Epoch 49/50
    938/938 [==============================] - 17s 18ms/step - loss: 0.4180 - acc: 0.9383 - val_loss: 0.7606 - val_acc: 0.8851
    Epoch 50/50
    938/938 [==============================] - 16s 17ms/step - loss: 0.4160 - acc: 0.9384 - val_loss: 0.7626 - val_acc: 0.8848





    <keras.callbacks.History at 0x7f76ea062760>



## 모델 테스트하기


```python
encoder_model = Model(inputs = encoder_inputs, outputs = encoder_states)
encoder_model.summary()
```

    Model: "model_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, None)]            0         
    _________________________________________________________________
    embedding (Embedding)        (None, None, 256)         1155584   
    _________________________________________________________________
    masking (Masking)            (None, None, 256)         0         
    _________________________________________________________________
    lstm (LSTM)                  [(None, 256), (None, 256) 525312    
    =================================================================
    Total params: 1,680,896
    Trainable params: 1,680,896
    Non-trainable params: 0
    _________________________________________________________________



```python
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = Embedding(fra_vocab_size, 256)(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state = decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]

decoder_outputs2 = decoder_softmax_layer(decoder_outputs2)
```


```python
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs2] + decoder_states2)
decoder_model.summary()
```

    Model: "model_2"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None, None)]       0                                            
    __________________________________________________________________________________________________
    embedding_2 (Embedding)         (None, None, 256)    1859328     input_2[0][0]                    
    __________________________________________________________________________________________________
    input_3 (InputLayer)            [(None, 256)]        0                                            
    __________________________________________________________________________________________________
    input_4 (InputLayer)            [(None, 256)]        0                                            
    __________________________________________________________________________________________________
    lstm_1 (LSTM)                   [(None, None, 256),  525312      embedding_2[0][0]                
                                                                     input_3[0][0]                    
                                                                     input_4[0][0]                    
    __________________________________________________________________________________________________
    dense (Dense)                   (None, None, 7263)   1866591     lstm_1[1][0]                     
    ==================================================================================================
    Total params: 4,251,231
    Trainable params: 4,251,231
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
eng2idx = eng_tokenizer.word_index
fra2idx = fra_tokenizer.word_index
idx2eng = eng_tokenizer.index_word
idx2fra = fra_tokenizer.index_word

```


```python
def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    states_value = encoder_model.predict(input_seq)

    # <SOS>에 해당하는 원-핫 벡터 생성
    target_seq = np.zeros((1,1))
    target_seq[0, 0] = fra2idx['<start>']
    
    stop_condition = False
    decoded_sentence = ""

    # stop_condition이 True가 될 때까지 루프 반복
    while not stop_condition:
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 예측 결과를 문자로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = idx2fra[sampled_token_index]

        # 현재 시점의 예측 문자를 예측 문장에 추가
        decoded_sentence += ' '+sampled_char

        # <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_char == '<end>' or
           len(decoded_sentence) > max_fra_seq_len):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence
```


```python
# 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2src(input_seq):
    temp=''
    for i in input_seq:
        if(i!=0):
            temp = temp + idx2eng[i]+' '
    return temp
```


```python
# 번역문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2tar(input_seq):
    temp=''
    for i in input_seq:
        if((i!=0 and i!=fra2idx['<start>']) and i!=fra2idx['<end>']):
            temp = temp + idx2fra[i] + ' '
    return temp
```


```python
for seq_index in [1,100,301,777,2222]:
    input_seq = encoder_input_test[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print(35 * "-")
    print('입력 문장:', seq2src(encoder_input_test[seq_index]))
    print('정답 문장:', seq2tar(decoder_input_test[seq_index]))
    print('번역기가 번역한 문장:', decoded_sentence[:len(decoded_sentence)-1])
```

    -----------------------------------
    입력 문장: take it . 
    정답 문장: prenez le ! 
    번역기가 번역한 문장:  qu . . . . ! . ! 
    -----------------------------------
    입력 문장: i have cabin fever . 
    정답 문장: je me sens comme un lion en cage . 
    번역기가 번역한 문장:  j me de . . . . 
    -----------------------------------
    입력 문장: get the camera . 
    정답 문장: prenez l appareil photo . 
    번역기가 번역한 문장:  prends la . . . 
    -----------------------------------
    입력 문장: i borrow money . 
    정답 문장: j emprunte de l argent . 
    번역기가 번역한 문장:  j ce mon de de d 
    -----------------------------------
    입력 문장: anything goes here . 
    정답 문장: ici tout est permis . 
    번역기가 번역한 문장:  l c c regardez quell



```python

```
