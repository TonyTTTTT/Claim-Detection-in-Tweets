import pandas as pd


dataset = 'CLEF2022'

if dataset == 'CLEF2021':
    train_path = '../clef2021-checkthat-lab/task1/data/subtask-1a--english/v1/dataset_train_v1_english.tsv'
    dev_path = '../clef2021-checkthat-lab/task1/data/subtask-1a--english/v1/dataset_dev_v1_english.tsv'
    # use dev set as testing set to compare with winner's performance
    test_path = '../clef2021-checkthat-lab/task1/test-gold/subtask-1a--english/subtask-1a--english/' \
                     'dataset_test_english.tsv'
elif dataset == 'CLEF2022':
    train_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1A_checkworthy/' \
                      'CT22_english_1A_checkworthy_train.tsv'
    dev_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1A_checkworthy/' \
                    'CT22_english_1A_checkworthy_dev.tsv'
    test_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/test/' \
                     'CT22_english_1A_checkworthy_test_gold.tsv'
elif dataset == 'CLEF20221b':
    train_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1B_claim/' \
                      'CT22_english_1B_claim_train.tsv'
    dev_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1B_claim/' \
                    'CT22_english_1B_claim_dev.tsv'
    test_path = '../clef2022-checkthat-lab/task1/data/subtasks-english/test/' \
                     'CT22_english_1B_claim_test_gold.tsv'


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


train_data = pd.read_csv(train_path, sep='\t', dtype=str)
dev_data = pd.read_csv(dev_path, sep='\t', dtype=str)
test_data = pd.read_csv(test_path, sep='\t', dtype=str)

train_data[train_data.columns[-1]] = train_data[train_data.columns[-1]].astype(int)
dev_data[dev_data.columns[-1]] = dev_data[dev_data.columns[-1]].astype(int)
test_data[test_data.columns[-1]] = test_data[test_data.columns[-1]].astype(int)


train_ids, train_topic_ids, train_texts, train_labels = read_df_to_lists(train_data)
dev_ids, dev_topic_ids, dev_texts, dev_labels = read_df_to_lists(dev_data)
test_ids, test_topic_ids, test_texts, test_labels = read_df_to_lists(test_data)


def convert_to_GPT_finetune_format(texts, labels):
    for i in range(0, len(labels)):
        if labels[i] == 0:
            labels[i] = 'no'
        else:
            labels[i] = 'yes'

    for i in range(0, len(texts)):
        texts[i] += "\ncheckworthy:"

    return texts, labels


train_texts_GPT, train_labels_GPT = convert_to_GPT_finetune_format(train_texts, train_labels)
test_texts_GPT, test_labels_GPT = convert_to_GPT_finetune_format(test_texts, test_labels)

train_out = pd.DataFrame(list(zip(train_texts_GPT, train_labels_GPT)), columns=['prompt', 'completion'])
test_out = pd.DataFrame(list(zip(test_texts_GPT, test_labels_GPT)), columns=['prompt', 'completion'])

train_out.to_csv('finetuned_GPT/{}_train_gpt_finetune_format.tsv'.format(dataset), sep='\t', index=False)
test_out.to_csv('finetuned_GPT/{}_test_gpt_finetune_format.tsv'.format(dataset), sep='\t', index=False)