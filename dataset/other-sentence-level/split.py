import pandas as pd

dataset_twitter = pd.read_csv('../LESA/Twitter.csv', encoding='utf-8')
dataset_twitter = dataset_twitter[['tweet_text', 'claim']]
dataset_twitter['claim'] = dataset_twitter['claim'].astype(float)
dataset_twitter = dataset_twitter.sample(frac=1, random_state=0).reset_index(drop=True)

## SPLIT INTO TRAIN AND TEST
split_ratio = 0.15
split_idx = int(dataset_twitter.shape[0] * (1 - split_ratio))

twitter_train = dataset_twitter.iloc[:split_idx]
twitter_test = dataset_twitter.iloc[split_idx:, :]

twitter_train.columns = ['text', 'claim']
twitter_test.columns = ['text', 'claim']

## DOWNSAMPLE TRAIN SET IN 1:1 RATIO
count_1_values, count_0_values = twitter_train['claim'].value_counts()

class_0 = twitter_train[twitter_train['claim'] == 0]
class_1 = twitter_train[twitter_train['claim'] == 1]

count_1_needed = int(count_0_values * 1)

class_1_under = class_1.sample(count_1_needed)

twitter_train = pd.concat([class_1_under, class_0], axis=0)

twitter_train['claim'] = twitter_train['claim'].astype(int)
twitter_test['claim'] = twitter_test['claim'].astype(int)

topics = ['LESA' for i in range(0, twitter_train.shape[0])]
tweet_ids = [i for i in range(0, twitter_train.shape[0])]

twitter_train.insert(0, 'tweet_id', tweet_ids)
twitter_train.insert(0, 'topic', topics)

topics = ['LESA' for i in range(0, twitter_test.shape[0])]
tweet_ids = [i+twitter_train.shape[0] for i in range(0, twitter_test.shape[0])]

twitter_test.insert(0, 'tweet_id', tweet_ids)
twitter_test.insert(0, 'topic', topics)

twitter_train.to_csv('LESA_train.tsv', sep='\t', index=False)
twitter_test.to_csv('LESA_test.tsv', sep='\t', index=False)
