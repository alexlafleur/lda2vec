from gensim.models.word2vec import Word2Vec



#polluter_tweets = "/home/alexandra/Social_Bots/Dataset/social_honeypot_icwsm_2011/content_polluters_tweets.txt"
#legitimate_tweets = "/home/alexandra/Social_Bots/Dataset/social_honeypot_icwsm_2011/legitimate_users_tweets.txt"

#poll_tweets = load_data(fn=polluter_tweets, delimiter="\t", usecols=2)
#leg_tweets = load_data(fn=legitimate_tweets, delimiter="\t", usecols=2)
#all_tweets = poll_tweets + leg_tweets

#tweets_fn = "tweets_shuffled.csv"
#tweets = util.load_data_user_history(fn=tweets_fn, delimiter="\t")
#print("data loaded")

import util
import pickle, numpy as np

id="honeypot_clean_model_revised"
polluter_tweets = "../lda2vec/tweets_shuffled_no_links.csv"
grouped_df = util.load_data_user_history(fn=polluter_tweets, delimiter="\t",names=["user_id","tweet_id", "tweet","created_at", "tweet_no_links"])

print("data loaded")
sentences = util.clean_sentences_user_history(grouped_df)
print("generated sentences")

#with open("sentences_no_stopwords_revised.txt", "wb") as fp:   #Pickling
#    pickle.dump(sentences, fp)
half = len(sentences)/2
np.save("sentences_no_stopwords_revised_1", sentences[:half])
np.save("sentences_no_stopwords_revised_2", sentences[half:])

model = Word2Vec(sentences=sentences, size=300, window=5, min_count=1, workers=4)
model.save(id)
print("model saved")

