from emoji import demojize
from nltk.tokenize import TweetTokenizer
from model_config import delete_at, delete_hashtag, delete_url, delete_emoji


tokenizer = TweetTokenizer()


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
        
        
def delToken(token):
    lowercased_token = token.lower()

    if (lowercased_token.startswith("http") or lowercased_token.startswith("www")) and delete_url:
        return ""
    elif len(token)<=2 and len(demojize(token))>2 and delete_emoji:
        return ""
    else:
        return token


# if do_demojize=False del all emoji
def normalizeTweet(tweets):
    print('del @: {}, del #: {}, del url: {}, del emoji: {}'.format(delete_at, delete_hashtag, delete_url, delete_emoji))
    if type(tweets) != list:
        tweets = [tweets]

    tweets_norm = []
    for tweet in tweets:
        tweet_norm = tweet.replace("’", "'").replace("…", "...")
        if delete_hashtag:
            tweet_norm = tweet_norm.replace("#", "")
        if delete_at:
            tweet_norm = tweet_norm.replace("@", "")
        tokens = tokenizer.tokenize(tweet_norm)

        tweet_norm = " ".join([delToken(token) for token in tokens])

        tweets_norm.append(" ".join(tweet_norm.split()))
    return tweets_norm


if __name__ == "__main__":
    example = "Vaccines  💵work by triggering a response in a person's immune system. That means some people will feel a little sore, tired or unwell after their #COVID19 vaccination. Most side effects are mild and should not last longer than a @MrGG. More on the vaccine: https://t.co/cSCb40c2mt #AmericanRescuePlan."
    r = normalizeTweet([
        # "I just voted YES for: 💵$1400 direct relief, payments 🏛Relief funding for state and local governments 💉More COVID-19 vaccines 🍽Funds to help struggling restaurants #AmericanRescuePlan.",
        # 'Netanyahu announcing Israel\'s new extreme measures against #corona. Orders Israelis to stop shaking hands, suggests following Indian custom of namaste instead. 🙏🏽 <link>',
        # "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier",
        # "This is a http://totally/shit/show said @MrGG!",
        # "India's gift of 100,000 COVID-19 vaccines arrived Barbados earlier today. This was a very special moment for all Barbadians and I want to thank Prime Minister Modi for his quick, decisive, and magnanimous action in allowing us to be the beneficiary of these vaccines. https://t.co/cSCb40c2mt",
        example,
    ])
