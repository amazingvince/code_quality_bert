# code_quality_bert

# Multi-Head Classification Model Inference

This README explains how to use the multi-head classification model for inference.

## Setup

First, ensure you have the necessary dependencies installed:

```bash
pip install transformers torch numpy
```

## Model Definition

The multi-head classification model is defined as follows:

```python
from transformers import AutoModel, AutoConfig
from torch import nn

class MultiHeadClassificationModel(nn.Module):
    def __init__(self, config, num_labels_list):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.name_or_path)
        self.num_labels_list = num_labels_list
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, num_labels) for num_labels in num_labels_list])

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits
```

## Loading the Model

To load the model and tokenizer:

```python
from transformers import AutoTokenizer
import torch

model_name = "HuggingFaceTB/fineweb-edu-scorer"  # Replace with your model's name
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
num_labels_list = [6, 6, 6]  # Adjust this based on your model's configuration
model = MultiHeadClassificationModel(config, num_labels_list)
model.load_state_dict(torch.load(f"{model_name}/pytorch_model.bin"))
model.eval()
```

## Inference Function

Here's a function to process text and get scores:

```python
import numpy as np

def get_scores(text):
    inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    scores = []
    for logits in outputs:
        score = logits.squeeze().float().numpy()
        scores.append(score)
    
    return scores
```

## Usage Example

Here's how to use the model for inference:

```python
text = "This is a test sentence."
scores = get_scores(text)

result = {
    "text": text,
    "scores": [score.tolist() for score in scores],
    "int_scores": [np.argmax(score) for score in scores]
}
print(result)
```

The output will look something like this:

```python
{
    'text': 'This is a test sentence.',
    'scores': [[0.1, 0.2, 0.3, 0.1, 0.2, 0.1], [0.2, 0.3, 0.1, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.3, 0.2, 0.1]],
    'int_scores': [2, 1, 3]
}
```

In this example, we have three heads, each providing a score for 6 classes. The `int_scores` represent the predicted class for each head.

## Note

Remember to adjust the `num_labels_list` to match your model's configuration. Also, make sure to replace `"HuggingFaceTB/fineweb-edu-scorer"` with the actual name or path of your saved model.
