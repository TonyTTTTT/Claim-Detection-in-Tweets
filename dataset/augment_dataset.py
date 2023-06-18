import pandas as pd


article_level_dataset_path = 'CheckThatLab2022-1b/CLEF_CB'
sentence_level_dataset_path = 'other-sentence-level/OC_processed/OC'

set_types = ['train', 'dev', 'test']
for set_type in set_types:
    df1 = pd.read_csv('{}_{}.tsv'.format(article_level_dataset_path, set_type), sep='\t', dtype=str)
    df2 = pd.read_csv('{}_{}.tsv'.format(sentence_level_dataset_path, set_type), sep='\t', dtype=str)
    # df1 = df1.drop(['tweet_url'], axis=1)

    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_combined['tweet_id'] = [i for i in range(0, df_combined.shape[0])]
    df_combined.to_csv('CheckThatLab2022-1b/CLEF_CB_OC_{}.tsv'.format(set_type), sep='\t', index=False)
