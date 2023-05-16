import pandas as pd
from sklearn.model_selection import train_test_split

df_processed = pd.DataFrame()
df = pd.read_csv('combined_sorted.csv')
df_processed.insert(0, 'topic', ['ClaimBuster' for i in range(0, df.shape[0])])
df_processed.insert(1, 'tweet_id', df['Sentence_id'])
df_processed.insert(2, 'text', df['Text'])

claim = []
for i in range(0, df['Verdict'].shape[0]):
    if df['Verdict'][i] == -1:
        claim.append(0)
    elif df['Verdict'][i] == 0 or df['Verdict'][i] == 1:
        claim.append(1)

df_processed.insert(3, 'claim', claim)

df_processed.to_csv('ClaimBuster.tsv', sep='\t', index=False)

train_set, test_set = train_test_split(df_processed, test_size=0.1, random_state=17)

train_set.to_csv('ClaimBuster_train.tsv', sep='\t', index=False)
test_set.to_csv('ClaimBuster_test.tsv', sep='\t', index=False)
