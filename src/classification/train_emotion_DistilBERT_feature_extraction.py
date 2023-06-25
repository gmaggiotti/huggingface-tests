from datasets import list_datasets, load_dataset
import pandas as pd

all_datasets = list_datasets()

emotions = load_dataset("emotion")
train_ds = emotions["train"]
#
# emotions.set_format(type="pandas")
# df = emotions["train"][:]
#
#
# def label_int2str(row):
#     return emotions["train"].features["label"].int2str(row)


# df["label_name"] = df["label"].apply(label_int2str)

# Look at the labels distribution
import matplotlib.pyplot as plt

# df["label_name"].value_counts(ascending=True).plot.barh()
# plt.title("Frequency of Classes")
# plt.show()


# Distribution of tweets length
# df["Words Per Tweet"] = df["text"].str.split().apply(len)
# df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False, color="black")
# plt.suptitle("")
# plt.xlabel("")
# plt.show()


emotions.reset_format()

# hide_output
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

import torch
from transformers import AutoModel

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(torch.device(device))


def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# extract hidden states of the whole batch
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

import numpy as np

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
X_train.shape, X_valid.shape

# train a Logistic Regression using the hidden states as inputs
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)

# eval
print(lr_clf.score(X_valid, y_valid))

import pickle
output = open('emotion_DistilBERT_logi_model.pkl', 'wb')
pickle.dump(lr_clf, output)
output.close()

hidden_states = X_valid, y_valid
output = open('emotion_DistilBERT_logi_hidden.pkl', 'wb')
pickle.dump(hidden_states, output)
output.close()

print("EOF")
