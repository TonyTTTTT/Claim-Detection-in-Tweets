import pandas as pd
import torch.utils.data
from transformers import AutoTokenizer
from TweetNormalizer import normalizeTweet
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data_preprocess_methods import *
import pickle
from model_config import model_path




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


class DataLoader:
    def __init__(self, preprocess_function, dataset):
        self.preprocess_function = preprocess_function
        self.train_dataset = []
        self.dev_dataset = []
        self.test_dataset = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, normalization=False)

        if dataset == 'CLEF2021':
            self.train_path = '../clef2021-checkthat-lab/task1/data/subtask-1a--english/v1/dataset_train_v1_english.tsv'
            self.dev_path = '../clef2021-checkthat-lab/task1/data/subtask-1a--english/v1/dataset_dev_v1_english.tsv'
            # use dev set as testing set to compare with winner's performance
            self.test_path = '../clef2021-checkthat-lab/task1/test-gold/subtask-1a--english/subtask-1a--english/' \
                             'dataset_test_english.tsv'
        elif dataset == 'CLEF2022':
            self.train_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1A_checkworthy/' \
                              'CT22_english_1A_checkworthy_train.tsv'
            self.dev_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1A_checkworthy/' \
                              'CT22_english_1A_checkworthy_dev.tsv'
            self.test_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/test/' \
                                'CT22_english_1A_checkworthy_test_gold.tsv'
        elif dataset == 'CLEF20221b':
            self.train_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1B_claim/' \
                              'CT22_english_1B_claim_train.tsv'
            self.dev_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1B_claim/' \
                            'CT22_english_1B_claim_dev.tsv'
            self.test_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/test/' \
                             'CT22_english_1B_claim_test_gold.tsv'

        self.read_data(dataset)

    @staticmethod
    def read_df_to_lists(data):
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

    def read_data(self, dataset):
        train_data = pd.read_csv(self.train_path, sep='\t')
        dev_data = pd.read_csv(self.dev_path, sep='\t')
        test_data = pd.read_csv(self.test_path, sep='\t')

        train_ids, train_topic_ids, train_texts_raw, train_labels = self.read_df_to_lists(train_data)
        dev_ids, dev_topic_ids, dev_texts_raw, dev_labels = self.read_df_to_lists(dev_data)
        test_ids, test_topic_ids, test_texts_raw, test_labels = self.read_df_to_lists(test_data)

        train_texts_raw = normalizeTweet(train_texts_raw)
        dev_texts_raw = normalizeTweet(dev_texts_raw)
        test_texts_raw = normalizeTweet(test_texts_raw)

        train_ids, train_topic_ids, train_texts, train_labels = self.preprocess_function(train_ids, train_topic_ids, train_texts_raw, train_labels, dataset+'_train')
        dev_ids, dev_topic_ids, dev_texts, dev_labels = self.preprocess_function(dev_ids, dev_topic_ids, dev_texts_raw, dev_labels, dataset+'_dev')
        test_ids, test_topic_ids, test_texts, test_labels = self.preprocess_function(test_ids, test_topic_ids, test_texts_raw, test_labels, dataset+'_test')

        if model_path == "vinai/bertweet-covid19-base-uncased":
            # if model_max_length < 64 or > 128, it will occur error when training with bertweet, maybe the reason is
            # the max_lenght of bertweet
            self.tokenizer.model_max_length = 64

            train_encodings = self.tokenizer(train_texts, truncation=True, padding='max_length')
            dev_encodings = self.tokenizer(dev_texts, truncation=True, padding='max_length')
            test_encodings = self.tokenizer(test_texts, truncation=True, padding='max_length')
        else:
            # self.tokenizer.model_max_length = 256
            #
            # train_encodings = self.tokenizer(train_texts, truncation=True, padding='max_length')
            # dev_encodings = self.tokenizer(dev_texts, truncation=True, padding='max_length')
            # test_encodings = self.tokenizer(test_texts, truncation=True, padding='max_length')

            train_encodings = self.tokenizer(train_texts, truncation=True, padding='longest')
            dev_encodings = self.tokenizer(dev_texts, truncation=True, padding='longest')
            test_encodings = self.tokenizer(test_texts, truncation=True, padding='longest')

        self.train_dataset = CheckThatDataset(train_encodings, train_ids, train_topic_ids, train_labels)
        self.dev_dataset = CheckThatDataset(dev_encodings, dev_ids, dev_topic_ids, dev_labels)
        self.test_dataset = CheckThatDataset(test_encodings, test_ids, test_topic_ids, test_labels)

    def get_dataset(self, include_test=False):
        if include_test:
            return self.train_dataset, self.dev_dataset, self.test_dataset

        return self.train_dataset, self.dev_dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    confusionMatrix = confusion_matrix(labels, preds)
    # PR = precision_recall_curve(labels, preds)
    acc = accuracy_score(labels, preds)
    # ROC = roc_curve(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': confusionMatrix.tolist(),
        # 'PR': PR,
        # 'ROC': ROC
    }


if __name__ == '__main__':
    dataset = 'CLEF2022'
    dataloader = DataLoader(preprocess_function=concate_all_frames, dataset=dataset)
    train_dataset, dev_dataset, test_dataset = dataloader.get_dataset(include_test=True)