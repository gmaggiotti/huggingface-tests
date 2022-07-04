import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings

warnings.filterwarnings('ignore')
SEQUENCE_LENGTH = 128
ROW_SIZE = 1200

df = pd.read_csv('../dataset/model_v1_predictions_utterance.csv', delimiter=',')
df.loc[df.ground_truth == "POSITIVE", "ground_truth"] = 1
df.loc[df.ground_truth == "NEGATIVE", "ground_truth"] = 0
df1 = df[df.ground_truth != "NEUTRAL"]
df2 = df1[['text', 'ground_truth']]
df2 = df2[:ROW_SIZE]
# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Want BERT instead of distilBERT? Uncomment the following line:
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# tokenize the list of sentences
tokenized = df2.text.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# now we can handle the tokenized senteces with Distilbert

# def add_padding(tokenized):
#     max_len = 0
#     for sentence in tokenized.values:
#         if len(sentence) > max_len:
#             max_len = len(sentence)
#     return np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])


def add_padding(tokenized):
    array = []
    for sentence in tokenized.values:
        if len(sentence) > SEQUENCE_LENGTH:
            array.append(sentence[:SEQUENCE_LENGTH])
        else:
            array.append(sentence + [0] * (SEQUENCE_LENGTH - len(sentence)))
    return np.array(array)


padded = add_padding(tokenized)

# Create a vector with attention mask and then run each sentence through BERT
attention_mask = np.where(padded != 0, 1, 0)
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# extract features:
#  Let's slice only the part of the output that we need. That is the output corresponding the first token of each
#  sentence. The way BERT does sentence classification, is that it adds a token called [CLS] (for classification)
#  at the beginning of every sentence. The output corresponding to that token can be thought of as an embedding
#  for the entire sentence.
features = last_hidden_states[0][:, 0, :].numpy()
labels = df2.ground_truth.values.tolist()

#  Split in training and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

# parameters = {'C': np.linspace(0.0001, 100, 20)}
# grid_search = GridSearchCV(LogisticRegression(), parameters)
# grid_search.fit(train_features, train_labels)
#
# print('best parameters: ', grid_search.best_params_)
# print('best scrores: ', grid_search.best_score_)

# train and return the mean accuracy of the given test set
# lr_clf = LogisticRegression(C=grid_search.best_params_['C'])
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
score = lr_clf.score(test_features, test_labels)
print("mean accuracy on the given test data and labels", score)

# Validation

df_with_neu = df[['text', 'ground_truth']]
validation_set = df_with_neu.sample(frac=0.5, random_state=200)[:ROW_SIZE]
val_tokenized = validation_set.text.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
val_padded = add_padding(tokenized)

# Create a vector with attention mask and then run each sentence through BERT
val_attention_mask = np.where(val_padded != 0, 1, 0)
val_input_ids = torch.tensor(val_padded)
val_attention_mask = torch.tensor(val_attention_mask)
with torch.no_grad():
    val_last_hidden_states = model(val_input_ids, attention_mask=val_attention_mask)

val_features = val_last_hidden_states[0][:, 0, :].numpy()
val_labels = validation_set.ground_truth.values.tolist()

score = lr_clf.predict_proba(val_features)

f_score = []
f_labels = []
for s, label in zip(score,val_labels):
    if label != "NEUTRAL":
        f_score.append(s[1])
        f_labels.append(label)
        print("Pos", s[1], label)

from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

display = PrecisionRecallDisplay.from_predictions(f_labels, f_score, name="LinearSVC")
_ = display.ax_.set_title("2-class Precision-Recall curve")
plt.show()
print('EOF')
