import pandas as pd
import json


# df_dev = pd.read_json('dev.json')
# df_test = pd.read_json('test.json')

with open('all_sents.json') as f:
    dict_all = json.load(f)

with open('dev.json') as f:
    dict_test= json.load(f)

with open('test.json') as f:
    dict_train= json.load(f)

df_train = pd.DataFrame()
text_train = []
claim_train = []
for key in dict_train.keys():
    for key2 in dict_train[key].keys():
        text_train.append(dict_train[key][key2]['sentence'])
        claim_train.append(1)
        dict_all.get(key).__delitem__(key2)

df_test = pd.DataFrame()
text_test = []
claim_test = []
for key in dict_test.keys():
    for key2 in dict_test[key].keys():
        text_test.append(dict_test[key][key2]['sentence'])
        claim_test.append(1)
        dict_all.get(key).__delitem__(key2)

text_all = []
claim_all = []
for key in dict_all.keys():
    for key2 in dict_all[key].keys():
        text_all.append(dict_all[key][key2]['sentence'])
        claim_all.append(0)

split_idx = int(len(text_all) * (len(text_train) / (len(text_train)+len(text_test))))

text_train.extend(text_all[:split_idx])
claim_train.extend(claim_all[:split_idx])

text_test.extend(text_all[split_idx:])
claim_test.extend(claim_all[split_idx:])

df_train.insert(0, 'claim', claim_train)
df_train.insert(0, 'text', text_train)
df_train.insert(0, 'tweet_id', [i for i in range(0, len(claim_train))])
df_train.insert(0, 'topic', ['NewsClaims' for i in range(0, len(claim_train))])

df_test.insert(0, 'claim', claim_test)
df_test.insert(0, 'text', text_test)
df_test.insert(0, 'tweet_id', [i+len(claim_train) for i in range(0, len(claim_test))])
df_test.insert(0, 'topic', ['NewsClaims' for i in range(0, len(claim_test))])

df_train.to_csv('NewsClaims_train.tsv', sep='\t', index=False)
df_test.to_csv('NewsClaims_test.tsv', sep='\t', index=False)