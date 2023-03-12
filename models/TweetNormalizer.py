from emoji import demojize
from nltk.tokenize import TweetTokenizer


tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) <= 2:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return demojize(token)
    else:
        return token
        
        
def delToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return ""
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return ""
    elif len(token) == 1:
        return ""
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweets, do_demojize=True):
    if type(tweets) != list:
        tweets = [tweets]

    tweets_norm = []
    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([normalizeToken(token) for token in tokens])
        # normTweet = " ".join([delToken(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't", " not")
            .replace("n 't", " not")
            .replace("ca n't", "can not")
            .replace("ai n't", "am not")
        )
        normTweet = (
            normTweet.replace("'m", " am")
            .replace("'re", " are")
            .replace("'s", " is")
            .replace("'ll", " will")
            .replace("'d", " had")
            .replace("'ve", " have")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )
        tweets_norm.append(" ".join(normTweet.split()))
    return tweets_norm


# if do_demojize=False del all emoji
def normalizeTweet_old(tweets):
    if type(tweets) != list:
        tweets = [tweets]

    tweets_norm = []
    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([normalizeToken(token) for token in tokens])
        # normTweet = " ".join([delToken(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )
        tweets_norm.append(" ".join(normTweet.split()))
    return tweets_norm


if __name__ == "__main__":
        r = normalizeTweet([
            'Netanyahu announcing Israel\'s new extreme measures against #corona. Orders Israelis to stop shaking hands, suggests following Indian custom of namaste instead. 🙏🏽 <link>',
            # "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier",
            # "This is a http://totally/shit/show said @MrGG!",
            "Vaccines work by triggering a response in a person's immune system. That means some people will feel a little sore, tired or unwell after their #COVID19 vaccination. Most side effects are mild and should not last longer than a week. More on the vaccine: HTTPURL"
        ])
