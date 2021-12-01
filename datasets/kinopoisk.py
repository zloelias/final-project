import glob
from datasets import concatenate_datasets, load_dataset

neg = glob.glob("./data/kinopoisk/dataset/neg/*")
neu = glob.glob("./data/kinopoisk/dataset/neu/*")
pos = glob.glob("./data/kinopoisk/dataset/pos/*")

kinopoisk_dataset1 = load_dataset('text', data_files=neg)
kinopoisk_dataset1 = kinopoisk_dataset1.map(lambda x: {'text': x['text'], 'labels': 2, 'label_name': 'negative'})

kinopoisk_dataset2 = load_dataset('text', data_files=neu)
kinopoisk_dataset2 = kinopoisk_dataset2.map(lambda x: {'text': x['text'], 'labels': 0, 'label_name': 'neutral'})

kinopoisk_dataset3 = load_dataset('text', data_files=pos)
kinopoisk_dataset3 = kinopoisk_dataset3.map(lambda x: {'text': x['text'], 'labels': 1, 'label_name': 'positive'})

kinopoisk_dataset = concatenate_datasets([kinopoisk_dataset1['train'], kinopoisk_dataset2['train'], kinopoisk_dataset3['train']])

kinopoisk_dataset = kinopoisk_dataset.train_test_split(test_size=0.1, train_size=0.9)


kinopoisk_dataset.push_to_hub(repo_id='zloelias/kinopoisk-reviews')

