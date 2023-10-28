import pickle
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
import torch
from transformers import AutoModel

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(torch.device(device))

text = "I feel in love"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")

inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)

outputs.last_hidden_state[:, 0].cpu().numpy()

# pkl_file = open('emotion_DistilBERT_logi_hidden.pkl', 'rb')
# hidden_states = pickle.load(pkl_file) i
# X_valid, y_valid = hidden_states

# train a Logistic Regression using the hidden states as inputs
pkl_file = open('emotion_DistilBERT_logi_model.pkl', 'rb')
lr_clf = pickle.load(pkl_file)

hidden_state = outputs.last_hidden_state[:, 0].cpu().numpy()

print(lr_clf.predict(hidden_state))

print("EOF")
