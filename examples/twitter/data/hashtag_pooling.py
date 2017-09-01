import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_hashtags(tweet):
    import re
    hashtags = re.findall(r"#(\w+)", tweet)
    return hashtags

def count_values(hashtag_tweets_dict):
    counts = {}
    for key, val in hashtag_tweets_dict.items():
        if len(val) >= 1000:
            counts[key] = len(val)
    return counts

def create_hashtag_dict(tweets):
    hashtag_tweets_dict = {}
    no_hashtags = []
    for tweet in tweets:
        if not pd.isnull(tweet):
            hashtags = extract_hashtags(tweet)
            if len(hashtags) == 0:
                no_hashtags.append(tweet)
                continue
            for hashtag in hashtags:
                if hashtag in hashtag_tweets_dict:
                    tweet_list = hashtag_tweets_dict[hashtag]
                    tweet_list.append(tweet)
                    hashtag_tweets_dict[hashtag] = tweet_list
                else:
                    tweet_list = [tweet]
                    hashtag_tweets_dict[hashtag] = tweet_list
    return hashtag_tweets_dict, no_hashtags


def get_hashtags_from_tfidf(tfidf, all_tweets, hashtag_dict ,threshold = 0.8):
    from sklearn.metrics.pairwise import linear_kernel
    print("Iterating over "+str(len(all_tweets))+" tweets.")
    for idx in range(0, len(all_tweets)):
        if idx % 1000==0:
            print("TFIDF on Index ",idx)
        if (not "#" in str(tweets[idx])):
            new_tweet = tfidf[idx:idx+1]
            # we have to infer a hashtag according to related tweets
            cosine_similarities = linear_kernel(new_tweet, tfidf).flatten()
            related_docs_indices = cosine_similarities.argsort()[:-5:-1]
            for i in related_docs_indices:
                if cosine_similarities[i] > threshold:
                    related_tweet = all_tweets[i]
                    if "#" in related_tweet:
                        hashtags = extract_hashtags(related_tweet)
                        print("hashtags for ",all_tweets[idx],"are\t", hashtags, "\tsimilarity:",cosine_similarities[i])
                        for hashtag in hashtags:
                            tweets_list = hashtag_dict[hashtag]
                            tweets_list.append(new_tweet)
                            hashtag_dict[hashtag] = tweets_list
    return hashtag_dict


def get_dict_values_list(dict):
    hashtag_labeled_tweets = []
    for hashtag_tweet in dict.values():
        for tweet in hashtag_tweet:
            hashtag_labeled_tweets.append(tweet)
    return hashtag_labeled_tweets

df = data=pd.read_csv("../lda2vec/tweets_shuffled_no_links.csv", delimiter="\t", names=["userid","tweetid","tweet","created_at","tweet_no_links"], dtype=str)
tweets = df.tweet_no_links.values
hashtag_tweets_dict, no_hashtags = create_hashtag_dict(tweets)
print(len(hashtag_tweets_dict))
hashtag_tweets_list = get_dict_values_list(hashtag_tweets_dict)
print("Non hashtag tweets: ",len(no_hashtags))
hashtags_tweets_dict_clone = hashtag_tweets_dict.copy()
all_tweets = no_hashtags + hashtag_tweets_list
tfidf = TfidfVectorizer().fit_transform(all_tweets)
hashtags_tweets_dict_clone = get_hashtags_from_tfidf(tfidf, all_tweets, hashtags_tweets_dict_clone)
print(len(hashtags_tweets_dict_clone))

print("saving the dictionary.")
import pickle
with open("hashtag_tweet_dict.pkl", "wb") as fp:   #Pickling
    pickle.dump(hashtags_tweets_dict_clone, fp, pickle.HIGHEST_PROTOCOL)