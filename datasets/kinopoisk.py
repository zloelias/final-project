import glob
from datasets import concatenate_datasets, load_dataset, Dataset

neg = glob.glob("./data/kinopoisk/dataset/neg/*")
neu = glob.glob("./data/kinopoisk/dataset/neu/*")
pos = glob.glob("./data/kinopoisk/dataset/pos/*")

negative = []


def clearRating(s):
    for i in range(10):
        s = s.replace(str(i)+' из 10', '')
    return s

i = 0
for file in neg:
    with open(file, 'r') as f:
        text = clearRating(f.read())
        negative.append({'text': text, 'labels': 2, 'label_name': 'negative'})
        i += 1
        #print('--------')
        #print(data[-1])
print(f'negative :{i}')

neutral = []
i = 0
for file in neu:
    with open(file, 'r') as f:
        neutral.append({'text': clearRating(f.read()), 'labels': 0, 'label_name': 'neutral'})
        i += 1
        #print('--------')
        #print(data[-1])
print(f'neutral :{i}')

positive = []
i = 0
for file in pos:
    with open(file, 'r') as f:
        positive.append({'text': clearRating(f.read()), 'labels': 1, 'label_name': 'positive'})
        i += 1
        #print('--------')
        #print(data[-1])
print(f'positive :{i}')

import pandas as pd

negative = pd.DataFrame(negative).sample(len(negative)*2, replace=True)
negative.dropna()

neutral = pd.DataFrame(neutral).sample(len(neutral)*2, replace=True)
neutral.dropna()

positive = pd.DataFrame(positive).sample(len(positive)//2, replace=True)
positive.dropna()

print('--------------------------------')
print('After resampling:')
print(f'negative: {negative.shape[0]}')
print(f'neutral: {neutral.shape[0]}')
print(f'positive: {positive.shape[0]}')

data = [negative, neutral, positive]
df = pd.concat(data)


kinopoisk_dataset = Dataset.from_pandas(df)
kinopoisk_dataset = kinopoisk_dataset.train_test_split(test_size=0.1, train_size=0.9)
kinopoisk_dataset.push_to_hub(repo_id='zloelias/kinopoisk-reviews')

