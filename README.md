# how-to-convert-sbert-models-to-onnx

Sentence-transformers library does not seem to provide tools to convert the sbert model ([sentence-transformers on huggingface.co/models](https://huggingface.co/sentence-transformers)) to an onnx model directly. And the SentenceTransformer class does not inherit from torch.nn.Module, which means that a sbert model is not a pytorch Module, so converting from a sbert model to an onnx model becomes more difficult. But we all know that there are many tutorials teaching us how to convert a pytorch model to an onnx model. So if we found how to convert the sbert model to a pytorch model, we can do a "sbert model -> pytorch model -> onnx model" converting. 

Here are the steps:

## 1. Download the model you need
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'sentence-transformers/distiluse-base-multilingual-cased-v1',  # for example
    device='cpu',
    cache_folder='any-specified-path')
```
You will get a folder with the following structure.
```
any-specified-path/
|-- sentence-transformers_distiluse-base-multilingual-cased-v1/
    |-- 1_Pooling/
    |-- 2_Dense/
    |-- modules.json
    [...other Transformer net files...]
```

## 2. Check the modules in the file modules.json
For the example above,
```json
[
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  },
  {
    "idx": 2,
    "name": "2",
    "path": "2_Dense",
    "type": "sentence_transformers.models.Dense"
  }
]
```

## 3. Create a pytorch model
```python
# torch_model.py
from sentence_transformers.models import Transformer, Pooling, Dense  # found in modules.json
import torch.nn as nn


class Distiluse(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer.load('any-specified-path/sentence-transformers/distiluse-base-multilingual-cased-v1')
        self.pooling = Pooling.load('any-specified-path/sentence-transformers/distiluse-base-multilingual-cased-v1/1_Pooling')
        self.dense = Dense.load('any-specified-path/sentence-transformers/distiluse-base-multilingual-cased-v1/2_Dense')

    def forward(self, features):
        outputs = self.transformer(features)
        outputs = self.pooling(outputs)
        outputs = self.dense(outputs)
        return outputs
```

## 4. Check whether the inferencing result has not been changed
```python
import numpy as np
import torch

from torch_model import Distiluse
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


def sbert_model(text):
    model = SentenceTransformer(r'any-specified-path/sentence-transformers/distiluse-base-multilingual-cased-v1')
    result = model.encode(text, batch_size=1)
    return result


def torch_model(text):
    tokenizer = AutoTokenizer.from_pretrained(r'any-specified-path/sentence-transformers/distiluse-base-multilingual-cased-v1')
    model = Distiluse()
    model.eval()
    encodings = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        result = model(encodings)
    result = result.sentence_embedding.numpy()
    result = np.squeeze(result, axis=0)
    return result


text = 'any sentence to test'
result_1 = sbert_model(text)
result_2 = torch_model(text)
print(np.allclose(result_1, result_2))  # shall be True
```

## 5. Finally
Now, we can do a "pytorch model -> onnx model" converting.
