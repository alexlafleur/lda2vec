
def test_word_2vec():
    from gensim.models.keyedvectors import KeyedVectors

    id = "honeypot_clean_model_revised"
    mymodel = KeyedVectors.load(id)
    n_dim = mymodel.wv.syn0.shape[1]
    print(n_dim)

    print(mymodel.wv.most_similar("today"))


test_word_2vec()

def test_tokenizer():
    import util

    f = open("nltk_tokens.txt", "w+")

    polluter_tweets = "content_polluters_tweets.txt"
    tweets = util.load_data(fn=polluter_tweets, delimiter="\t", usecols=2)
    sentences = util.preprocess_data(tweets)

    dicti = {}
    for tokens in sentences:
        for token in tokens:
            if not token in dicti:
                dicti[token]=True

    for key in dicti.keys():
        f.write(key.encode('utf-8') + "\n")


def test_tokenizer_spacy(limit=500000):
    import util
    f = open("nltk_tokens_spacy~.txt", "w+")

    polluter_tweets = "content_polluters_tweets.txt"
    tweets = util.load_data(fn=polluter_tweets, delimiter="\t",usecols=2)
    tokens, vocab= util.preprocess_spacy_tokenize(tweets[:limit])
    print("done tokenizing")

    for key,value in vocab.items():
        print(key,value)
        f.write(value.encode("utf-8") + "\n")




#test_tokenizer()
#test_tokenizer_spacy(limit=10)
import util
import pandas as pd
import numpy as np

"""
df  = pd.read_csv("content_polluters_tweets.txt", delimiter="\t", header=None, names=["user_id","tweet_id", "tweet","created_at"])
df["tweet_no_links"]=df["tweet"].apply(lambda t:util.remove_links(t))
df.to_csv("content_polluters_tweets_no_links.txt", sep="\t", header=None, index=False)


df  = pd.read_csv("legitimate_users_tweets.txt", delimiter="\t", header=None, names=["user_id","tweet_id", "tweet","created_at"], dtype=str)
print("dataframe loaded")
df["tweet_no_links"]=df["tweet"].apply(lambda t:util.remove_links(t))
print("removed links")
df.to_csv("legitimate_users_tweets_no_links.txt", sep="\t", header=None, index=False)
print("dataframe saved")
#print(util.tokenizeText(u"These colours are manifold! I can't believe it..."))

"""
