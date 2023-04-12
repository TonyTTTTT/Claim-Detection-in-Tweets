import pandas as pd

origin_dataset = pd.read_csv('../clef2022-checkthat-lab/task1/data/subtasks-english/CT22_english_1A_checkworthy/'
                     'CT22_english_1A_checkworthy_train.tsv', sep='\t')
origin_dataset = origin_dataset.drop('tweet_url', axis=1)

rewrite_method = 'explain_by_GPT_100_words'
rewrite_dataset = pd.read_csv('preprocess_datasets_tsv/CLEF2022_train_' + rewrite_method + '.tsv', sep='\t')

frames = [origin_dataset, rewrite_dataset]
concatenated_dataset = pd.concat(frames)

concatenated_dataset.to_csv('preprocess_datasets_tsv/CLEF2022_train_{}_augmented.tsv'.format(rewrite_method),
                            sep='\t', index=False)
