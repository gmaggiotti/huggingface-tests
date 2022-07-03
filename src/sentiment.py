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

df = pd.read_csv('../dataset/train.tsv', delimiter='\t', header=None)

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Want BERT instead of distilBERT? Uncomment the following line:
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

batch_1 = df[:2000]
# tokenize the list of sentences
tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# now we can handle the tokenized senteces with Distilbert

def add_padding(tokenized):
    max_len = 0
    for sentence in tokenized.values:
        if len(sentence) > max_len:
            max_len = len(sentence)
    return np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])


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
labels = batch_1[1]

#  Split in training and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

parameters = {'C': np.linspace(0.0001, 100, 20)}
grid_search = GridSearchCV(LogisticRegression(), parameters)
grid_search.fit(train_features, train_labels)

print('best parameters: ', grid_search.best_params_)
print('best scrores: ', grid_search.best_score_)

# train and return the mean accuracy of the given test set
# lr_clf = LogisticRegression(C=grid_search.best_params_['C'])
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
score = lr_clf.score(test_features, test_labels)
print("mean accuracy on the given test data and labels", score)

from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt


display = PrecisionRecallDisplay.from_estimator(
    lr_clf, test_features, test_labels, name="LinearSVC"
)
_ = display.ax_.set_title("2-class Precision-Recall curve")

plt.show()
print('EOF')
