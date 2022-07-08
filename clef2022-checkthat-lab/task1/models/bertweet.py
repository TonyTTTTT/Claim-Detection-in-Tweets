import torch
from transformers import AutoModel, AutoTokenizer
from TweetNormalizer import normalizeTweet

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

# For transformers v4.x+:
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)

# For transformers v3.x:
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# INPUT TWEET IS ALREADY NORMALIZED!
line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"
line2 = 'DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier 😢'

input_ids = torch.tensor([tokenizer.encode(line)])
input_ids2 = torch.tensor([tokenizer.encode(line2)])
input_ids3 = torch.tensor([tokenizer.encode(normalizeTweet(line2))])

with torch.no_grad():
    features = bertweet(input_ids)  # Models outputs are now tuples

# With TensorFlow 2.0+:
# from transformers import TFAutoModel
# bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")