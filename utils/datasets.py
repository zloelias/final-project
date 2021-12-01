def count_classes(dataset, label_key='labels'):
    classes = set()
    dataset['train'].map(lambda x: classes.add(x[label_key]))
    return len(classes)