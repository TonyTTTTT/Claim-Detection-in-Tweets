import string
from emoji import demojize
from nltk.tokenize import TweetTokenizer
from model_config import delete_at, delete_hashtag, delete_url, delete_emoji, delete_tail, replace_covid, \
    delete_tail_punc, replace_user, recover_punc, delete_punc
import nltk
import nltk.data


tokenizer = TweetTokenizer()
sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) <= 2:
        return demojize(token)
    else:
        return token
        
        
def delToken(token, user_idx):
    lowercased_token = token.lower()

    if (lowercased_token.startswith("http") or lowercased_token.startswith("www")) and delete_url:
        return "", user_idx
    elif token.startswith("@") and replace_user:
        user_idx += 1
        return "USER{}".format(user_idx-1), user_idx
    elif len(token)<=2 and len(demojize(token))>2 and delete_emoji:
        return "", user_idx
    elif ('covid' in lowercased_token or 'corona' in lowercased_token) and replace_covid:
        return "Ebola", user_idx
    else:
        return token, user_idx


# if do_demojize=False del all emoji
def normalizeTweet(tweets):
    print('del tail hashtag: {}, del @: {}, del #: {}, del url: {}, del emoji: {}, replace covid: {}, del tail '
          'punctuation: {}, replace user: {}, recover punc: {}, del punc: {}'.format(delete_tail, delete_at,
                                                                                     delete_hashtag, delete_url,
                                                                                     delete_emoji, replace_covid,
                                                                                     delete_tail_punc, replace_user,
                                                                                     recover_punc, delete_punc))
    if type(tweets) != list:
        tweets = [tweets]

    tweets_norm = []
    for tweet in tweets:
        tweet_norm = tweet.replace("’", "'").replace("…", "...")

        if delete_tail:
            sentences = sentence_splitter.tokenize(tweet_norm)
            if sentences != []:
                last_sen = tokenizer.tokenize(sentences[-1])
                del_last_sen = True
                for token in last_sen:
                    if not token.startswith('#') and not token.startswith('http') and not token.startswith('@'):
                        del_last_sen = False
                        break
                if del_last_sen:
                    new_last_sen = []
                    for token in last_sen:
                        if not token.startswith('#') and not token.startswith('@'):
                            new_last_sen.append(token)
                    if new_last_sen != []:
                        sentences[-1] = " ".join(new_last_sen)
                    else:
                        sentences.pop()
                tweet_norm = " ".join(sentences)

        tokens = tokenizer.tokenize(tweet_norm)

        if delete_tail_punc:
            if tokens[-1] in string.punctuation:
                tokens.pop()

        tweet_norm = ""
        user_idx = 0
        for token in tokens:
            res_token, user_idx = delToken(token, user_idx)
            tweet_norm = tweet_norm + " " + res_token
        # tweet_norm = " ".join([delToken(token) for token in tokens])

        if delete_hashtag:
            tweet_norm = tweet_norm.replace("#", "")
        if delete_at:
            tweet_norm = tweet_norm.replace("@", "")
        if recover_punc:
            for punc in string.punctuation:
                tweet_norm = tweet_norm.replace(" {}".format(punc), "{}".format(punc))
        if delete_punc:
            for punc in string.punctuation:
                tweet_norm = tweet_norm.replace("{}".format(punc), "")


        tweets_norm.append(" ".join(tweet_norm.split()))
    return tweets_norm


if __name__ == "__main__":
    examples = [
        "@aaa I got covid, @bbb shit. @ccc",
        ".",
        "Vaccines  💵work by triggering a response in a person's immune system. That means some people will feel a little sore, tired or unwell after their #COVID19 vaccination. Most side effects are mild and should not last longer than a @MrGG. More on the vaccine: https://t.co/cSCb40c2mt #AmericanRescuePlan #AAA #cat #dog",
        "This is how you annihilate northern Italy—check the date of the post! The CCP and WHO basically killed the Chinese, and then let it spread to hide their bioweapons work. #CCPVirus #CCPLiedPeopleDied #CCPVirus @abc #CCP_is_terrorist #Coronavirustruth #COVID19 https://t.co/D49H4d7oET",
        "Prime Minister of Israel Benjamin Netanyahu  @netanyahu  encourages Israelis to adopt the Indian way of greeting #Namaste at a press conference to mitigate the spread of #coronavirus"
        ]

    r = normalizeTweet(examples)
