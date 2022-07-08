import pandas as pd


def count_same_id(df1, df2):
    same_id_tweet_ids = []
    diff_label_ids = []
    for id in df1['tweet_id'].values:
        if id in df2['tweet_id'].values:
            same_id_tweet_ids.append(id)
            try:
                df1_label = df1.loc[df1['tweet_id'] == id]['class_label'].item()
                df2_label = df2.loc[df2['tweet_id'] == id]['class_label'].item()
                if df1_label != df2_label:
                    diff_label_ids.append(id)
            except:
                print("No Label")

    return same_id_tweet_ids, diff_label_ids


def count_same_text(df1, df2):
    same_text_tweet_ids = []
    for data in df1.iterrows():
        text = data[1]['tweet_text']
        if text in df2['tweet_text'].values:
            same_text_tweet_ids.append(data[1]['tweet_id'])

    return same_text_tweet_ids


def compare(df1, df2, df1_name, df2_name):
    print('Comparing {}(len: {}) and {}(len: {})'.format(df1_name, len(df1), df2_name, len(df2)))
    same_id_tweet_ids, diff_label_ids = count_same_id(df1, df2)
    print("Same tweets id: {}".format(len(same_id_tweet_ids)))

    same_text_tweet_ids = count_same_text(df1, df2)
    print("Same tweets text: {}".format(len(same_text_tweet_ids)))

    diff_text_ids = list(set(same_id_tweet_ids) - set(same_text_tweet_ids))
    print("Same id, different text: {}".format(len(diff_text_ids)))
    print("Same id, different label: {}".format(len(diff_label_ids)))
    print("==============================")


if __name__ == '__main__':
    d1 = {'tweet_id': [0, 1, 2, 3, 4, 5], 'tweet_text': ['Dog', 'Cat', 'Cow', 'Pig', 'Eagle', 'Egg'],
          'class_label': [0, 0, 0, 0, 0, 0]}
    d2 = {'tweet_id': [0, 2, 4, 6, 9], 'tweet_text': ['Dog', 'Cow', 'eagle', 'Duck', 'Goose'],
          'class_label': [1, 1, 1, 1, 1]}
    test1_df1 = pd.DataFrame(data=d1)
    test1_df2 = pd.DataFrame(data=d2)
    compare(test1_df1, test1_df2, 'test1_df1', 'test1_df2')  # should be 3, 2, 1, 3

    task_1a_train_2021_path = 'clef2021-checkthat-lab/task1/data/subtask-1a--english/v1/dataset_train_v1_english.tsv'
    task_1a_dev_2021_path = 'clef2021-checkthat-lab/task1/data/subtask-1a--english/v1/dataset_dev_v1_english.tsv'
    task_1a_test_2021_path = 'clef2021-checkthat-lab/task1/test-gold/subtask-1a--english/subtask-1a--english/dataset_test_english.tsv'

    task_1a_train_2022_path = 'clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1A_checkworthy/CT22_english_1A_checkworthy_train.tsv'
    task_1a_dev_2022_path = 'clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1A_checkworthy/CT22_english_1A_checkworthy_dev.tsv'
    task_1a_devtest_2022_path = 'clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1A_checkworthy/CT22_english_1A_checkworthy_dev_test.tsv'
    task_1a_test_2022_path = 'clef2022-checkthat-lab/task1/data/subtasks-english/test/CT22_english_1A_checkworthy_test.tsv'

    task_1a_train_2021 = pd.read_csv(task_1a_train_2021_path, sep='\t', dtype=str)
    task_1a_dev_2021 = pd.read_csv(task_1a_dev_2021_path, sep='\t', dtype=str)
    task_1a_test_2021 = pd.read_csv(task_1a_test_2021_path, sep='\t', dtype=str)

    task_1a_train_2022 = pd.read_csv(task_1a_train_2022_path, sep='\t', dtype=str)
    task_1a_dev_2022 = pd.read_csv(task_1a_dev_2022_path, sep='\t', dtype=str)
    task_1a_devtest_2022 = pd.read_csv(task_1a_devtest_2022_path, sep='\t', dtype=str)
    task_1a_test_2022 = pd.read_csv(task_1a_test_2022_path, sep='\t', dtype=str)

    sets = [[task_1a_train_2021, 'task_1a_train_2021'], [task_1a_dev_2021, 'task_1a_dev_2021'],
            [task_1a_test_2021, 'task_1a_test_2021'], [task_1a_train_2022, 'task_1a_train_2022'],
            [task_1a_dev_2022, 'task_1a_dev_2022'], [task_1a_devtest_2022, 'task_1a_devtest_2022'],
            [task_1a_test_2022, 'task_1a_test_2022']]

    for i in range(0, len(sets)):
        for j in range(i + 1, len(sets)):
            compare(sets[i][0], sets[j][0], sets[i][1], sets[j][1])

    # compare(task_1a_train_2021, task_1a_train_2022, 'task_1a_train_2021', 'task_1a_train_2022')
    # compare(task_1a_dev_2021, task_1a_train_2022, 'task_1a_dev_2021', 'task_1a_dev_2022')
