import string
from emoji import demojize
from nltk.tokenize import TweetTokenizer


tokenizer = TweetTokenizer()


def calculate_special_symbol(tweets, labels):
    claim_cnt = labels.count(1)
    non_claim_cnt = labels.count(0)

    special_sym_cnt_claim = 0
    special_sym_cnt_non_claim = 0
    special_symbols = ['http', 'www', '@', '#']

    for i in range(0, len(tweets)):
        tweet_norm = tweets[i].replace("’", "'").replace("…", "...")
        tokens = tokenizer.tokenize(tweet_norm)
        for token in tokens:
            lowercased_token = token.lower()
            for special_symbol in special_symbols:
                if lowercased_token.startswith(special_symbol):
                    if labels[i] == 1:
                        special_sym_cnt_claim += 1
                    else:
                        special_sym_cnt_non_claim += 1

            if len(token)<=2 and len(demojize(token))>2:
                if labels[i] == 1:
                    special_sym_cnt_claim += 1
                else:
                    special_sym_cnt_non_claim += 1


    avg_special_sym_cnt_claim = special_sym_cnt_claim / claim_cnt
    avg_special_sym_cnt_non_claim = special_sym_cnt_non_claim / non_claim_cnt

    print('avg_special_sym_cnt_claim: {}\navg_special_sym_cnt_non_claim: {}'.format(avg_special_sym_cnt_claim, avg_special_sym_cnt_non_claim))