import pandas as pd

dataset_name = 'LESA'
origin_dataset = pd.read_csv('../dataset/LESA/{}_train.tsv'.format(dataset_name), sep='\t', dtype=str)
# origin_dataset = origin_dataset.drop('tweet_url', axis=1)

rewrite_method = 'rewrite_by_GPT_v7'
rewrite_dataset = pd.read_csv('preprocess_datasets_GPT/{}_{}_train.tsv'.format(dataset_name, rewrite_method), sep='\t', dtype=str)

frames = [origin_dataset, rewrite_dataset]
concatenated_dataset = pd.concat(frames)
concatenated_dataset['tweet_id'] = [i for i in range(0, concatenated_dataset.shape[0])]

concatenated_dataset.to_csv('preprocess_datasets_GPT/{}_{}_augmented.tsv'.format(dataset_name, rewrite_method),
                            sep='\t', index=False)
