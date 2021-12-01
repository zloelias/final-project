import datasets
from datasets import load_dataset

import pandas as pd

df = pd.read_csv('./data/lenta_ru/lenta-ru-train.csv')
df.dropna().to_csv('./data/lenta_ru/lenta-ru-train.csv', index=None)

lenta_dataset = load_dataset('csv', data_files='./data/lenta_ru/lenta-ru-train.csv')['train']
lenta_dataset = lenta_dataset.train_test_split(test_size=0.1, train_size=0.9)

lenta_dataset['train'].rename_column_('topic_label', 'labels')
lenta_dataset['test'].rename_column_('topic_label', 'labels')

lenta_dataset.push_to_hub(repo_id='zloelias/lenta-ru')