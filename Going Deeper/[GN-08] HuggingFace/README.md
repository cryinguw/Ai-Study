# HuggingFace 커스텀 프로젝트 만들기

## 1. 모듈 import


```python
import tensorflow as tf
import numpy as np
import pandas as pd
import os,sys,copy,time
import urllib

# Hugging face
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import datasets
from datasets import load_dataset
```


```python
train1 = os.getenv('HOME')+'/aiffel/ratings_train.txt'
test1 = os.getenv('HOME')+'/aiffel/ratings_test.txt'
```


```python
train = pd.read_table(train1)
test = pd.read_table(test1)
```


```python
train.head()
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
      <th>id</th>
      <th>document</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9976970</td>
      <td>아 더빙.. 진짜 짜증나네요 목소리</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3819312</td>
      <td>흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10265843</td>
      <td>너무재밓었다그래서보는것을추천한다</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9045019</td>
      <td>교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6483659</td>
      <td>사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
huggingface_nsmc_dataset = load_dataset('nsmc')
print(huggingface_nsmc_dataset)
```

    Using custom data configuration default
    Reusing dataset nsmc (/aiffel/.cache/huggingface/datasets/nsmc/default/1.1.0/bfd4729bf1a67114e5267e6916b9e4807010aeb238e4a3c2b95fbfa3a014b5f3)



      0%|          | 0/2 [00:00<?, ?it/s]


    DatasetDict({
        train: Dataset({
            features: ['id', 'document', 'label'],
            num_rows: 150000
        })
        test: Dataset({
            features: ['id', 'document', 'label'],
            num_rows: 50000
        })
    })


## 2. 토크나이저 생성


```python
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels = 2)
```

    Some weights of the model checkpoint at klue/bert-base were not used when initializing BertForSequenceClassification: ['cls.predictions.decoder.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.weight', 'cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at klue/bert-base and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.



```python
def transform(data):
  return tokenizer(
      data['document'],
      truncation = True,
      #padding = 'longest', #'max_length','longest', #'max_length',
      return_token_type_ids = False,
      )
  
examples = huggingface_nsmc_dataset['train'][:2]
examples_transformed = transform(examples)

print(examples)
print(examples_transformed)
```

    {'id': ['9976970', '3819312'], 'document': ['아 더빙.. 진짜 짜증나네요 목소리', '흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나'], 'label': [0, 1]}
    {'input_ids': [[2, 1376, 831, 2604, 18, 18, 4229, 9801, 2075, 2203, 2182, 4243, 3], [2, 1963, 18, 18, 18, 11811, 2178, 2088, 28883, 16516, 2776, 18, 18, 18, 18, 10737, 2156, 2015, 2446, 2232, 6758, 2118, 1380, 6074, 3]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}



```python
# 데이터 토크나이징 완료
encoded_dataset = huggingface_nsmc_dataset.map(transform, batched=True)

# 동적패딩: Bucketing 
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```


      0%|          | 0/150 [00:00<?, ?ba/s]



      0%|          | 0/50 [00:00<?, ?ba/s]



```python
from datasets import load_metric
metric = load_metric('glue', 'qnli')

def compute_metrics(eval_pred):    
    predictions,labels = eval_pred
    print(predictions.shape, predictions)
    predictions = np.argmax(predictions, axis=1)
    print(predictions.shape, predictions)
    return metric.compute(predictions=predictions, references = labels)
```

## 3. 훈련 및 평가


```python
# TrainingArguments 생성 : Trainer을 활용하는 형태로 모델 재생성
from transformers import Trainer, TrainingArguments
output_dir = os.getenv('HOME')+'/aiffel/hugging_face_transformers/output'
metric_name = 'accuracy'

training_arguments = TrainingArguments(
    output_dir, # output이 저장될 경로
    evaluation_strategy="steps",              # "epoch", #evaluation하는 빈도
    learning_rate = 0.001,                   #2e-5, #learning_rate   1e-3
    per_device_train_batch_size = 512,      #64, #64, #16, # 각 device 당 batch size    512
    per_device_eval_batch_size = 512,      #64, #64, #16, # evaluation 시에 batch size   512
    num_train_epochs = 2,                  # train 시킬 총 epochs   20
    weight_decay = 0.01,                   # weight decay
    save_strategy="epoch",                 # 저장은 epoch 마다
    metric_for_best_model=metric_name,  
)
```


```python
trainer = Trainer(
    model= model,                           # 학습시킬 model
    args=training_arguments,                  # TrainingArguments을 통해 설정한 arguments
    train_dataset=encoded_dataset['train'],    # training dataset
    eval_dataset=encoded_dataset['test'],       # test dataset
    compute_metrics=compute_metrics,
    data_collator = data_collator,
    tokenizer = tokenizer,
)
```


```python
trainer.evaluate(encoded_dataset['test'])
```

    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 512




<div>

  <progress value='98' max='98' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [98/98 04:58]
</div>



    (50000, 2) [[ 0.3536646   0.13695504]
     [ 0.3748738  -0.00793599]
     [ 0.4523907  -0.15237457]
     ...
     [-0.06852636  0.20821825]
     [ 0.37208036  0.02408335]
     [ 0.38024706  0.29182684]]
    (50000,) [0 0 0 ... 1 0 0]





    {'eval_loss': 0.6977577209472656,
     'eval_accuracy': 0.51222,
     'eval_runtime': 301.3266,
     'eval_samples_per_second': 165.933,
     'eval_steps_per_second': 0.325}




```python
# TrainingArguments 생성 : Trainer을 활용하는 형태로 모델 재생성
from transformers import Trainer, TrainingArguments
output_dir = os.getenv('HOME')+'/aiffel/hugging_face_transformers/output'
metric_name = 'accuracy'

training_arguments = TrainingArguments(
    output_dir, # output이 저장될 경로
    evaluation_strategy="steps",              # "epoch", #evaluation하는 빈도
    learning_rate = 2e-5,                    #learning_rate   1e-3
    per_device_train_batch_size = 16,      #64, #64, #16, # 각 device 당 batch size    512
    per_device_eval_batch_size = 16,      #64, #64, #16, # evaluation 시에 batch size   512
    num_train_epochs = 2,                  # train 시킬 총 epochs   20
    weight_decay = 0.01,                   # weight decay
    save_strategy="epoch",                 # 저장은 epoch 마다
    metric_for_best_model=metric_name,  
)
```

    using `logging_steps` to initialize `eval_steps` to 500
    PyTorch: setting up devices
    The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).



```python
trainer = Trainer(
    model= model,                           # 학습시킬 model
    args=training_arguments,                  # TrainingArguments을 통해 설정한 arguments
    train_dataset=encoded_dataset['train'],    # training dataset
    eval_dataset=encoded_dataset['test'],       # test dataset
    compute_metrics=compute_metrics,
    data_collator = data_collator,
    tokenizer = tokenizer,
)
```


```python
trainer.train()
```

    The following columns in the training set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running training *****
      Num examples = 150000
      Num Epochs = 2
      Instantaneous batch size per device = 16
      Total train batch size (w. parallel, distributed & accumulation) = 16
      Gradient Accumulation steps = 1
      Total optimization steps = 18750




    <div>

      <progress value='18750' max='18750' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [18750/18750 3:19:31, Epoch 2/2]
    </div>
    <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500</td>
      <td>0.382300</td>
      <td>0.316742</td>
      <td>0.865000</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>0.316600</td>
      <td>0.298721</td>
      <td>0.875000</td>
    </tr>
    <tr>
      <td>1500</td>
      <td>0.300200</td>
      <td>0.347037</td>
      <td>0.866260</td>
    </tr>
    <tr>
      <td>2000</td>
      <td>0.284400</td>
      <td>0.288353</td>
      <td>0.881920</td>
    </tr>
    <tr>
      <td>2500</td>
      <td>0.293100</td>
      <td>0.291645</td>
      <td>0.876720</td>
    </tr>
    <tr>
      <td>3000</td>
      <td>0.283900</td>
      <td>0.267559</td>
      <td>0.888980</td>
    </tr>
    <tr>
      <td>3500</td>
      <td>0.278000</td>
      <td>0.321031</td>
      <td>0.876940</td>
    </tr>
    <tr>
      <td>4000</td>
      <td>0.280700</td>
      <td>0.279730</td>
      <td>0.887620</td>
    </tr>
    <tr>
      <td>4500</td>
      <td>0.274100</td>
      <td>0.277605</td>
      <td>0.886080</td>
    </tr>
    <tr>
      <td>5000</td>
      <td>0.261300</td>
      <td>0.264133</td>
      <td>0.896860</td>
    </tr>
    <tr>
      <td>5500</td>
      <td>0.256000</td>
      <td>0.264503</td>
      <td>0.896740</td>
    </tr>
    <tr>
      <td>6000</td>
      <td>0.263300</td>
      <td>0.255161</td>
      <td>0.896560</td>
    </tr>
    <tr>
      <td>6500</td>
      <td>0.254300</td>
      <td>0.269955</td>
      <td>0.897560</td>
    </tr>
    <tr>
      <td>7000</td>
      <td>0.260500</td>
      <td>0.260875</td>
      <td>0.900200</td>
    </tr>
    <tr>
      <td>7500</td>
      <td>0.274000</td>
      <td>0.241241</td>
      <td>0.901020</td>
    </tr>
    <tr>
      <td>8000</td>
      <td>0.260200</td>
      <td>0.257818</td>
      <td>0.900160</td>
    </tr>
    <tr>
      <td>8500</td>
      <td>0.253000</td>
      <td>0.257117</td>
      <td>0.900600</td>
    </tr>
    <tr>
      <td>9000</td>
      <td>0.256500</td>
      <td>0.237378</td>
      <td>0.902140</td>
    </tr>
    <tr>
      <td>9500</td>
      <td>0.227400</td>
      <td>0.287444</td>
      <td>0.903620</td>
    </tr>
    <tr>
      <td>10000</td>
      <td>0.192400</td>
      <td>0.291255</td>
      <td>0.904320</td>
    </tr>
    <tr>
      <td>10500</td>
      <td>0.199000</td>
      <td>0.294434</td>
      <td>0.904440</td>
    </tr>
    <tr>
      <td>11000</td>
      <td>0.177000</td>
      <td>0.307879</td>
      <td>0.902680</td>
    </tr>
    <tr>
      <td>11500</td>
      <td>0.193400</td>
      <td>0.304167</td>
      <td>0.898340</td>
    </tr>
    <tr>
      <td>12000</td>
      <td>0.185600</td>
      <td>0.285408</td>
      <td>0.904280</td>
    </tr>
    <tr>
      <td>12500</td>
      <td>0.181300</td>
      <td>0.290567</td>
      <td>0.904100</td>
    </tr>
    <tr>
      <td>13000</td>
      <td>0.178800</td>
      <td>0.318203</td>
      <td>0.904400</td>
    </tr>
    <tr>
      <td>13500</td>
      <td>0.197400</td>
      <td>0.275849</td>
      <td>0.904360</td>
    </tr>
    <tr>
      <td>14000</td>
      <td>0.171600</td>
      <td>0.286664</td>
      <td>0.904800</td>
    </tr>
    <tr>
      <td>14500</td>
      <td>0.188100</td>
      <td>0.276537</td>
      <td>0.905140</td>
    </tr>
    <tr>
      <td>15000</td>
      <td>0.174300</td>
      <td>0.296438</td>
      <td>0.905340</td>
    </tr>
    <tr>
      <td>15500</td>
      <td>0.178800</td>
      <td>0.277625</td>
      <td>0.905460</td>
    </tr>
    <tr>
      <td>16000</td>
      <td>0.175800</td>
      <td>0.283747</td>
      <td>0.905080</td>
    </tr>
    <tr>
      <td>16500</td>
      <td>0.173500</td>
      <td>0.285223</td>
      <td>0.905880</td>
    </tr>
    <tr>
      <td>17000</td>
      <td>0.175600</td>
      <td>0.290372</td>
      <td>0.906300</td>
    </tr>
    <tr>
      <td>17500</td>
      <td>0.166700</td>
      <td>0.290834</td>
      <td>0.906800</td>
    </tr>
    <tr>
      <td>18000</td>
      <td>0.175700</td>
      <td>0.288977</td>
      <td>0.907580</td>
    </tr>
    <tr>
      <td>18500</td>
      <td>0.174600</td>
      <td>0.286714</td>
      <td>0.906760</td>
    </tr>
  </tbody>
</table><p>


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.3226981   1.5576096 ]
     [ 0.47015467  0.12922174]
     [-0.16868162 -0.86988676]
     ...
     [ 0.8387326  -1.1134475 ]
     [ 2.4216716  -2.767542  ]
     [ 1.3505646  -1.2666966 ]]
    (50000,) [1 0 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.194687    1.5615238 ]
     [ 0.4478849   0.02146624]
     [-0.03245687 -0.80478597]
     ...
     [ 0.24770191 -0.86740017]
     [ 2.3757906  -2.7244272 ]
     [ 0.49460283 -0.45395628]]
    (50000,) [1 0 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.5982176   1.8831034 ]
     [ 0.51068693 -0.14431214]
     [ 1.4019889  -1.9242096 ]
     ...
     [ 2.282714   -2.9479039 ]
     [ 3.119944   -3.2582142 ]
     [ 1.5118806  -1.6828741 ]]
    (50000,) [1 0 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.6380533   2.0347307 ]
     [ 0.3756262  -0.039908  ]
     [-0.17175047 -0.98410445]
     ...
     [ 0.63952744 -1.5587982 ]
     [ 2.7453902  -3.2448468 ]
     [ 0.23030525 -0.5460899 ]]
    (50000,) [1 0 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.7809902   2.3022854 ]
     [ 0.01993692  0.32855216]
     [-0.36667597 -0.21270259]
     ...
     [ 0.25963578 -0.7377296 ]
     [ 2.3413978  -2.8138046 ]
     [ 0.27618748 -0.28823334]]
    (50000,) [1 1 1 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.0279098   1.7240653 ]
     [ 0.07063915  0.22264719]
     [ 0.1558031  -0.7640131 ]
     ...
     [ 0.45037407 -1.021008  ]
     [ 2.3946934  -2.801915  ]
     [ 0.9005351  -0.94992405]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.3796384   2.3327792 ]
     [-0.48556557  1.0796227 ]
     [-0.33935088 -0.41027078]
     ...
     [ 0.09428083 -0.69850314]
     [ 3.0410097  -3.224456  ]
     [ 1.1150961  -1.3213358 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.6902586   1.9248221 ]
     [-0.26261452  0.7337987 ]
     [-0.32128575 -0.35336655]
     ...
     [ 0.32770562 -1.047807  ]
     [ 2.9421837  -3.1723409 ]
     [ 0.06829955 -0.04109202]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.1501994   2.7736313 ]
     [-0.11168169  0.51073444]
     [ 0.8819972  -1.5057081 ]
     ...
     [ 1.2785295  -1.8749967 ]
     [ 3.0901206  -3.2849255 ]
     [ 1.0922989  -0.8413526 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.9093903   2.54092   ]
     [-0.48958576  1.2931406 ]
     [ 0.13939212 -0.9613981 ]
     ...
     [ 0.44101337 -0.8155441 ]
     [ 2.9148898  -3.06254   ]
     [ 1.1386117  -0.8990127 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.1112401   2.5037355 ]
     [-0.24110657  0.75206757]
     [ 0.3457529  -1.4933429 ]
     ...
     [ 1.7682297  -2.5055962 ]
     [ 3.0211275  -3.2745883 ]
     [ 0.8028887  -0.99554545]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.6554493   1.8227468 ]
     [-0.29952914  0.7185249 ]
     [ 0.66576546 -1.7112707 ]
     ...
     [ 1.0167812  -1.8458428 ]
     [ 2.3423374  -2.8684928 ]
     [ 0.29713282 -0.7598111 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.8616731   2.5283155 ]
     [-0.22838373  0.75383013]
     [ 0.53713155 -1.5506343 ]
     ...
     [ 1.3379229  -2.1424932 ]
     [ 2.6819768  -3.1687756 ]
     [-0.11113081 -0.19777954]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.6418463   2.375093  ]
     [ 0.11145958  0.3532473 ]
     [ 0.79966813 -1.7486546 ]
     ...
     [ 0.50563276 -1.3980007 ]
     [ 2.7259874  -3.171922  ]
     [ 0.53259146 -0.91028124]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.4518179   2.0760083 ]
     [ 0.02258695  0.35604054]
     [ 0.4450873  -1.2586132 ]
     ...
     [-0.4843872  -0.11420755]
     [ 2.289786   -2.7472858 ]
     [ 0.8167892  -1.0940466 ]]
    (50000,) [1 1 0 ... 1 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.7106395   2.3355558 ]
     [-0.40344766  1.0613755 ]
     [ 0.9716669  -1.6602943 ]
     ...
     [ 0.5842584  -1.2507491 ]
     [ 2.9984672  -3.230086  ]
     [ 1.0158728  -1.0145298 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.9609308   2.6716597 ]
     [-0.3895571   0.84733224]
     [ 1.1033995  -2.0431955 ]
     ...
     [ 0.03028924 -0.83643186]
     [ 2.5532525  -3.1679943 ]
     [ 0.76262784 -1.0176846 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.7088133   2.3726826 ]
     [-0.39063868  0.95624083]
     [ 1.0127991  -1.7403146 ]
     ...
     [ 0.12422004 -0.9460077 ]
     [ 3.0438924  -3.3163    ]
     [ 1.3523451  -1.4982514 ]]
    (50000,) [1 1 0 ... 0 0 0]


    Saving model checkpoint to /aiffel/aiffel/hugging_face_transformers/output/checkpoint-9375
    Configuration saved in /aiffel/aiffel/hugging_face_transformers/output/checkpoint-9375/config.json
    Model weights saved in /aiffel/aiffel/hugging_face_transformers/output/checkpoint-9375/pytorch_model.bin
    tokenizer config file saved in /aiffel/aiffel/hugging_face_transformers/output/checkpoint-9375/tokenizer_config.json
    Special tokens file saved in /aiffel/aiffel/hugging_face_transformers/output/checkpoint-9375/special_tokens_map.json
    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.022908    2.9016116 ]
     [-0.51678807  1.212932  ]
     [ 1.8249114  -2.6563537 ]
     ...
     [ 0.12189508 -0.6925392 ]
     [ 3.29378    -3.4516442 ]
     [ 1.6245687  -1.733438  ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.8958389   2.7637146 ]
     [-0.74774677  1.5035726 ]
     [ 1.5194724  -2.4797711 ]
     ...
     [-0.14719819 -0.5951033 ]
     [ 3.2197847  -3.519077  ]
     [ 1.7048578  -2.0318184 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.1264453   2.9975152 ]
     [-0.92055106  1.737956  ]
     [ 1.7929804  -2.7875845 ]
     ...
     [ 0.4941606  -1.498339  ]
     [ 3.4791675  -3.7262087 ]
     [ 1.851873   -2.1137223 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.1467438   2.7223797 ]
     [-1.3257294   2.2359056 ]
     [ 2.2534168  -3.046104  ]
     ...
     [ 0.64295876 -1.4920814 ]
     [ 3.494222   -3.7308598 ]
     [ 1.8012785  -2.1498334 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.8847947   2.6775067 ]
     [-1.1397136   2.0428834 ]
     [ 1.3476651  -2.1883955 ]
     ...
     [-0.22036923 -0.56359446]
     [ 3.2715895  -3.6868136 ]
     [ 1.2724344  -1.5238366 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.764191    2.7201803 ]
     [-1.2264837   2.1787665 ]
     [ 1.9312259  -2.7300563 ]
     ...
     [-0.05618178 -0.6127234 ]
     [ 2.8842533  -3.4982097 ]
     [ 1.652776   -1.9292569 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.2972856   3.0626535 ]
     [-1.1015155   1.9687145 ]
     [ 2.0000372  -2.8308184 ]
     ...
     [ 0.21776578 -0.9879082 ]
     [ 3.2241154  -3.7005107 ]
     [ 1.6427761  -1.8970473 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.369203    3.0907516 ]
     [-1.0065601   1.8751143 ]
     [ 2.0937357  -2.9806547 ]
     ...
     [-0.39057752 -0.16225515]
     [ 3.217614   -3.757125  ]
     [ 1.4306653  -1.8700261 ]]
    (50000,) [1 1 0 ... 1 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-1.9141     2.4446373]
     [-0.5649489  1.3356127]
     [ 1.7041512 -2.6136131]
     ...
     [ 0.4455439 -1.3170937]
     [ 2.7694478 -3.4799132]
     [ 1.2189225 -1.6441227]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.189473    2.697199  ]
     [-0.40694252  1.1629691 ]
     [ 2.117797   -3.0372322 ]
     ...
     [ 0.34465608 -1.2718291 ]
     [ 3.1332128  -3.710242  ]
     [ 1.3482983  -1.9368354 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.0185196   2.5515916 ]
     [-0.30827725  1.0719876 ]
     [ 1.94718    -2.8642447 ]
     ...
     [ 0.3379473  -1.1744902 ]
     [ 3.3778365  -3.834376  ]
     [ 1.3300214  -1.7532945 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.3345997  2.9788966]
     [-0.5836303  1.4741081]
     [ 1.5563492 -2.496638 ]
     ...
     [-0.3077438 -0.3310492]
     [ 3.1326883 -3.737399 ]
     [ 1.340068  -1.7387664]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.192431    2.817046  ]
     [-0.5014699   1.4040213 ]
     [ 1.2744353  -2.1956346 ]
     ...
     [ 0.10037883 -0.88209885]
     [ 3.1952538  -3.731927  ]
     [ 1.1098621  -1.4267218 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.04883     2.6710126 ]
     [-0.45103186  1.357137  ]
     [ 1.2349644  -2.140392  ]
     ...
     [-0.38596842 -0.13536482]
     [ 3.2916007  -3.8306227 ]
     [ 1.2017089  -1.5255188 ]]
    (50000,) [1 1 0 ... 1 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.223267    2.8658054 ]
     [-0.43972856  1.3030032 ]
     [ 1.4713228  -2.3985822 ]
     ...
     [ 0.14785507 -0.92288053]
     [ 3.3259268  -3.851736  ]
     [ 1.0063154  -1.3391583 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.288857    2.9444666 ]
     [-0.47022042  1.34006   ]
     [ 1.7303284  -2.6347094 ]
     ...
     [ 0.75470483 -1.5630388 ]
     [ 3.3766418  -3.8763103 ]
     [ 1.2166256  -1.5762489 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.1810482   2.8590648 ]
     [-0.35694832  1.2162777 ]
     [ 1.688046   -2.5680888 ]
     ...
     [ 0.7517183  -1.5386033 ]
     [ 3.440646   -3.9195375 ]
     [ 1.2160867  -1.5384934 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.1263819   2.7993166 ]
     [-0.33590072  1.1990703 ]
     [ 1.7375814  -2.6101975 ]
     ...
     [ 0.5445172  -1.3227887 ]
     [ 3.377713   -3.902636  ]
     [ 1.2473344  -1.6138761 ]]
    (50000,) [1 1 0 ... 0 0 0]


    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16


    (50000, 2) [[-2.112761    2.7836838 ]
     [-0.40424395  1.2812573 ]
     [ 1.6378759  -2.5063622 ]
     ...
     [ 0.41083452 -1.167566  ]
     [ 3.3316545  -3.8665137 ]
     [ 1.1358777  -1.4947085 ]]
    (50000,) [1 1 0 ... 0 0 0]


    Saving model checkpoint to /aiffel/aiffel/hugging_face_transformers/output/checkpoint-18750
    Configuration saved in /aiffel/aiffel/hugging_face_transformers/output/checkpoint-18750/config.json
    Model weights saved in /aiffel/aiffel/hugging_face_transformers/output/checkpoint-18750/pytorch_model.bin
    tokenizer config file saved in /aiffel/aiffel/hugging_face_transformers/output/checkpoint-18750/tokenizer_config.json
    Special tokens file saved in /aiffel/aiffel/hugging_face_transformers/output/checkpoint-18750/special_tokens_map.json
    
    
    Training completed. Do not forget to share your model on huggingface.co/models =)
    
    





    TrainOutput(global_step=18750, training_loss=0.22986080993652344, metrics={'train_runtime': 11971.6652, 'train_samples_per_second': 25.059, 'train_steps_per_second': 1.566, 'total_flos': 9733020604318080.0, 'train_loss': 0.22986080993652344, 'epoch': 2.0})




```python
trainer.evaluate(encoded_dataset['test'])
```

    The following columns in the evaluation set  don't have a corresponding argument in `BertForSequenceClassification.forward` and have been ignored: document, id.
    ***** Running Evaluation *****
      Num examples = 50000
      Batch size = 16




<div>

  <progress value='3125' max='3125' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [3125/3125 03:36]
</div>



    (50000, 2) [[-2.0881035   2.7594452 ]
     [-0.3890107   1.2618679 ]
     [ 1.6213051  -2.4911911 ]
     ...
     [ 0.42239082 -1.1803601 ]
     [ 3.320626   -3.860013  ]
     [ 1.1174031  -1.4742037 ]]
    (50000,) [1 1 0 ... 0 0 0]





    {'eval_loss': 0.2848275899887085,
     'eval_accuracy': 0.90676,
     'eval_runtime': 216.6969,
     'eval_samples_per_second': 230.737,
     'eval_steps_per_second': 14.421,
     'epoch': 2.0}



## 4. 결론

Hugging Face 의 transformers를 사용하여, klue/bert_base model을 통해, NSMC 데이터를 Positive, Negative 감성분석을 실행해 보면서,Hugging Face 사용방법을 익혀보는 프로젝트

0.90676의 정학도를 보여줬다.
