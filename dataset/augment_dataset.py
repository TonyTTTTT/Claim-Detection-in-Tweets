import pandas as pd


article_level_dataset_path = 'LESA/LESA'
sentence_level_dataset_path = 'other-sentence-level/ClaimBuster_processed/ClaimBuster'

set_types = ['train', 'dev', 'test']
for set_type in set_types:
    df1 = pd.read_csv('{}_{}.tsv'.format(article_level_dataset_path, set_type), sep='\t', dtype=str)
    df2 = pd.read_csv('{}_{}.tsv'.format(sentence_level_dataset_path, set_type), sep='\t', dtype=str)

    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_combined['tweet_id'] = [i for i in range(0, df_combined.shape[0])]
    df_combined.to_csv('LESA/LESA_CB_{}.tsv'.format(set_type), sep='\t', index=False, quotechar='"', quoting=3, escapechar='\n')