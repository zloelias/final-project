import glob
from datasets import concatenate_datasets, load_dataset, Dataset

neg = glob.glob("./data/kinopoisk/dataset/neg/*")
neu = glob.glob("./data/kinopoisk/dataset/neu/*")
pos = glob.glob("./data/kinopoisk/dataset/pos/*")

data = []


def clearRating(s):
    for i in range(10):
        s = s.replace(str(i)+' из 10', '')
    return s

for file in neg:
    with open(file, 'r') as f:
        data.append({'text': clearRating(f.read()), 'labels': 2, 'label_name': 'negative'})
        #print('--------')
        #print(data[-1])

for file in neu:
    with open(file, 'r') as f:
        data.append({'text': clearRating(f.read()), 'labels': 0, 'label_name': 'neutral'})
        #print('--------')
        #print(data[-1])

for file in pos:
    with open(file, 'r') as f:
        data.append({'text': clearRating(f.read()), 'labels': 1, 'label_name': 'positive'})
        #print('--------')
        #print(data[-1])

import pandas as pd

df = pd.DataFrame(data).sample(frac=1)
df.dropna()

kinopoisk_dataset = Dataset.from_pandas(df)
kinopoisk_dataset = kinopoisk_dataset.train_test_split(test_size=0.1, train_size=0.9)
kinopoisk_dataset.push_to_hub(repo_id='zloelias/kinopoisk-reviews')

