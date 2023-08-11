from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
import pandas as pd
import numpy as np


file_name = 'CLEF2022_1b_zeroshot_by_GPT'
path = 'preprocess_datasets_GPT/{}_test.tsv'.format(file_name)
data = pd.read_csv(path, sep='\t', dtype=str)

preds = []
for pred in data['tweet_text'].values:
    if 'no' in pred.lower():
        preds.append(0)
    elif 'yes' in pred.lower():
        preds.append(1)
    else:
        preds.append(-1)

labels = data['class_label'].values.astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
f1_macro = f1_score(labels, preds, average='macro')
confusionMatrix = confusion_matrix(labels, preds)
agree = preds == labels
wrong_predicted_idx = np.where(agree == False)[0].tolist()
acc = accuracy_score(labels, preds)

print('acc: {}, f1: {}, macro f1: {}\nconfusion matrix: {}'.format(acc, f1, f1_macro, confusionMatrix))
