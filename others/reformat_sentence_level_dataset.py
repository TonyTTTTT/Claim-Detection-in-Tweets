import os
import pandas as pd


for path in os.listdir():
    if os.path.isdir(path) and 'processed' not in path:
        print(path)
        cnt = 0
        df_whole_dataset = pd.DataFrame()
        if not os.path.exists('{}_processed'.format(path)):
            os.makedirs('{}_processed'.format(path))

        for file_name in os.listdir(path):
            print(file_name)
            df = pd.read_csv(os.path.join(path, file_name), sep='\t', header=None, names=['text', 'claim'], index_col=None, quotechar='"', quoting=3)
            df.insert(0, 'topic', [path for i in range(0, df.shape[0])])
            df.insert(1, 'tweet_id', [cnt+i for i in range(0, df.shape[0])])
            cnt += df.shape[0]
            df.to_csv('{}_processed/{}_{}.tsv'.format(path, path, file_name[4:]), sep='\t', index=False, quotechar='"', quoting=3)
            df_whole_dataset = pd.concat([df_whole_dataset, df], ignore_index=True)

        df_whole_dataset.to_csv('{}_processed/{}.tsv'.format(path, path), sep='\t', index=False, quotechar='"', quoting=3)
        print('total data: {}'.format(cnt))
        print("====================")