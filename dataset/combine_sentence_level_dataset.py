import os
import pandas as pd


df_combined_train = pd.DataFrame()
df_combined_dev = pd.DataFrame()
df_combined_test = pd.DataFrame()
cnt_train = 0
cnt_dev = 0
cnt_test = 0

for path in os.listdir():
    if 'processed' in path:

        for file_name in os.listdir(path):
            print(file_name)
            df = pd.read_csv(os.path.join(path, file_name), sep='\t', index_col=None, quotechar='"', quoting=3)

            if 'train' in file_name:
                df['tweet_id'] = [cnt_train+i for i in range(0, df.shape[0])]
                cnt_train += df.shape[0]
                df_combined_train = pd.concat([df_combined_train, df], ignore_index=True)
            elif 'dev' in file_name:
                df['tweet_id'] = [cnt_dev + i for i in range(0, df.shape[0])]
                cnt_dev += df.shape[0]
                df_combined_dev = pd.concat([df_combined_dev, df], ignore_index=True)
            elif 'test' in file_name:
                df['tweet_id'] = [cnt_test + i for i in range(0, df.shape[0])]
                cnt_test += df.shape[0]
                df_combined_test = pd.concat([df_combined_test, df], ignore_index=True)

df_combined_train.to_csv('sentence_level_train.tsv', sep='\t', index=False, quotechar='"', quoting=3)
df_combined_dev.to_csv('sentence_level_dev.tsv', sep='\t', index=False, quotechar='"', quoting=3)
df_combined_test.to_csv('sentence_level_test.tsv', sep='\t', index=False, quotechar='"', quoting=3)

