# load packages
import torch
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('../dataset/train.tsv', delimiter='\t', header=None)

print('number of samples', df.shape)

texts = list(set(df.iloc[:,0]))

file_name = 'testing.txt'
with open(file_name, 'w') as f:
    f.write(" |EndOfText|\n".join(texts))

#
# python transformers/examples/pytorch//language-modeling/run_clm.py --model_name_or_path distilgpt2 --train_file dataset/testing.txt --do_train --num_train_epochs 3 --overwrite_output_dir --per_device_train_batch_size 2 --output_dir output
