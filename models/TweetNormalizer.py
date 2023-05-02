from emoji import demojize
from nltk.tokenize import TweetTokenizer


tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    # if token.startswith("@"):
    #     return "@USER"
    if lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) <= 2:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            # return demojize(token)
            if len(demojize(token)) > 2:
                return ""
            else:
                return token
    else:
        return token
        
        
def delToken(token):
    lowercased_token = token.lower()
    # if token.startswith("@"):
    #     return "@USER"
    if lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return ""
    elif len(token)<=2 and len(demojize(token))>2:
        return ""
    else:
        return token


def normalizeTweet_mine(tweets):
    if type(tweets) != list:
        tweets = [tweets]

    subject_word_list = ['He', 'he', 'She', 'she', 'It', 'it', 'That', 'that', 'There', 'there', 'I', 'They', 'they',
                         'This', 'this', 'These', 'these', 'Who', 'who', 'How', 'how', 'What', 'what', 'Where',
                         'where', 'When', 'when', 'Why', 'why']
    tweets_norm = []
    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([normalizeToken(token) for token in tokens])
        # normTweet = " ".join([delToken(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot", "can not")
            .replace("can't", "can not")
            .replace("ain't", "am not")
            .replace("n't", " not")
            .replace("n 't", " not")
        )

        for word in subject_word_list:
            normTweet = (
                normTweet.replace(word+"'s", word+" is")
            )

        normTweet = (
            normTweet.replace("'m", " am")
            .replace("'re", " are")
            .replace("'s", " 's")
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
def normalizeTweet(tweets):
    if type(tweets) != list:
        tweets = [tweets]

    tweets_norm = []
    for tweet in tweets:
        tweet_norm = tweet.replace("’", "'").replace("…", "...")
        # tokens = tokenizer.tokenize(tweet)
        # tokens = tweet.split(" ")
        # normTweet = " ".join([delToken(token) for token in tokens])
        # normTweet = normTweet.replace(" .", ".").replace(" ,", ",")

        # normTweet = (
        #     normTweet.replace("cannot ", "can not ")
        #     .replace("n't ", " n't ")
        #     .replace("n 't ", " n't ")
        #     .replace("ca n't", "can't")
        #     .replace("ai n't", "ain't")
        # )
        # normTweet = (
        #     normTweet.replace("'m ", " 'm ")
        #     .replace("'re ", " 're ")
        #     .replace("'s ", " 's ")
        #     .replace("'ll ", " 'll ")
        #     .replace("'d ", " 'd ")
        #     .replace("'ve ", " 've ")
        # )
        # normTweet = (
        #     normTweet.replace(" p . m .", "  p.m.")
        #     .replace(" p . m ", " p.m ")
        #     .replace(" a . m .", " a.m.")
        #     .replace(" a . m ", " a.m ")
        # )
        tweets_norm.append(" ".join(tweet_norm.split()))
    return tweets_norm


if __name__ == "__main__":
        r = normalizeTweet([
            # "I just voted YES for: 💵$1400 direct relief, payments 🏛Relief funding for state and local governments 💉More COVID-19 vaccines 🍽Funds to help struggling restaurants #AmericanRescuePlan.",
            # 'Netanyahu announcing Israel\'s new extreme measures against #corona. Orders Israelis to stop shaking hands, suggests following Indian custom of namaste instead. 🙏🏽 <link>',
            # "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier",
            # "This is a http://totally/shit/show said @MrGG!",
            # "India's gift of 100,000 COVID-19 vaccines arrived Barbados earlier today. This was a very special moment for all Barbadians and I want to thank Prime Minister Modi for his quick, decisive, and magnanimous action in allowing us to be the beneficiary of these vaccines. https://t.co/cSCb40c2mt",
            "Vaccines work by triggering a response in a person's immune system. That means some people will feel a little sore, tired or unwell after their #COVID19 vaccination. Most side effects are mild and should not last longer than a week. More on the vaccine: HTTPURL"
        ])
