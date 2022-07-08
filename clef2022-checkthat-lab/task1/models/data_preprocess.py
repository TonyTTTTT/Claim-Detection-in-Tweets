import pandas as pd
import torch.utils.data
from transformers import AutoTokenizer
from TweetNormalizer import normalizeTweet
import torch
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve, roc_curve
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 2021
# train_path = '../data/subtask-1a--english/v1/dataset_train_v1_english.tsv'
# dev_path = '../data/subtask-1a--english/v1/dataset_dev_v1_english.tsv'

# 2022
train_path = '../data/subtasks-english/CT22_english_1A_checkworthy/CT22_english_1A_checkworthy_train.tsv'
dev_path = '../data/subtasks-english/CT22_english_1A_checkworthy/CT22_english_1A_checkworthy_dev_test.tsv'
test_path = '../data/subtasks-english/test/CT22_english_1A_checkworthy_test.tsv'

train_data = pd.read_csv(train_path, sep='\t')
dev_data = pd.read_csv(dev_path, sep='\t')
test_data = pd.read_csv(test_path, sep='\t')


def read_data(data):
    texts = []
    labels = []
    ids = []
    topic_ids = []
    if data.columns[-1] != 'class_label' and data.columns[-1] != 'check_worthiness':
        for row in data.iterrows():
            # get the tweet_text field
            texts.append(row[1].values[3])
            # get teh check_worthiness field
            topic_ids.append(row[1].values[0])
            ids.append(row[1].values[1])
        return ids, topic_ids, texts

    for row in data.iterrows():
        # get the tweet_text field
        texts.append(row[1].values[3])
        # get teh check_worthiness field
        labels.append(row[1].values[-1])
        topic_ids.append(row[1].values[0])
        ids.append(row[1].values[1])
    return ids, topic_ids, texts, labels


train_ids, train_topic_ids, train_texts, train_labels = read_data(train_data)
dev_ids, dev_topic_ids, dev_texts, dev_labels = read_data(dev_data)
test_ids, test_topic_ids, test_texts = read_data(test_data)


# train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)
# train_ids, val_ids, train_topic_ids, val_topic_ids = train_test_split(train_ids, train_topic_ids, test_size=0.2)

# For transformers v4.x+:
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=False)
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-uncased", use_fast=False, normalization=False)
tokenizer.model_max_length = 64

train_texts = normalizeTweet(train_texts)
dev_texts = normalizeTweet(dev_texts)
test_texts = normalizeTweet(test_texts)

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length')
dev_encodings = tokenizer(dev_texts, truncation=True, padding='max_length')
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length')


class CheckThatDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, ids, topic_ids, labels=None):
        self.encodings = encodings
        self.labels = labels
        self.ids = ids
        self.topic_ids = topic_ids

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx])
        # item['id_num'] = self.ids[idx]
        # item['topic_id'] = self.topic_ids[idx]
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def get_dataset(include_test=False):
    train_dataset = CheckThatDataset(train_encodings, train_ids, train_topic_ids, train_labels) # deal with split train val
    dev_dataset = CheckThatDataset(dev_encodings, dev_ids, dev_topic_ids, dev_labels)
    if include_test:
        test_dataset = CheckThatDataset(test_encodings, test_ids, test_topic_ids)
        return train_dataset, dev_dataset, test_dataset

    return train_dataset, dev_dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    # PR = precision_recall_curve(labels, preds)
    acc = accuracy_score(labels, preds)
    # ROC = roc_curve(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        # 'PR': PR,
        # 'ROC': ROC
    }


if __name__ == '__main__':
    train_dataset, dev_dataset, test_dataset = get_dataset(include_test=True)