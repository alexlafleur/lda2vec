import nltk
import numpy as np

def load_data(fn, delimiter = "-$$$-", dtype="str", comments=None, usecols=0):
    """
        loads a csv file that contains textual tweets and decodes the texts. The csv is assummed to have texts in one column and only the delimiter.
        \n is not sufficient as delimiter, because tweets can contain breaklines.

    :param delimiter: delimiter to separate the textual data from other.
    :param dtype: the type of data. Usually str
    :param comments: to override the default of '#' which has sepcial meaning in twitter
    :param usecols: column number where to find the textual data
    :return: list of unicode texts
    """

    tweets = np.loadtxt(fname=fn, delimiter=delimiter, dtype=dtype, comments=comments, usecols=usecols)
    decoded_texts = []
    for text in tweets:
           decoded_text = text.decode('utf-8')
           decoded_texts.append(decoded_text)
    return decoded_texts


def load_data_user_history(fn, delimiter = "\t", header=None, names=["user_id","tweet_id", "tweet","created_at"], dtype=str):
    """
        loads a csv file that contains textual tweets and decodes the texts. The csv is assummed to have texts in one column and only the delimiter.
        \n is not sufficient as delimiter, because tweets can contain breaklines.

    :param delimiter: delimiter to separate the textual data from other.
    :param dtype: the type of data. Usually str
    :param comments: to override the default of '#' which has sepcial meaning in twitter
    :param usecols: column number where to find the textual data
    :return: list of unicode texts
    """
    import pandas as pd

    data = pd.read_csv(fn, delimiter=delimiter, header=header, names=names, dtype=dtype)
    return data.groupby(["user_id"])


def preprocess_data(data):
    """
        Preprocesses the raw data by removing links, tokenizing and ommiting tokens containing of only one character (punctuation, @, #, ...)

    :param data: list of texts
    :return: list of token lists
    """

    import re
    pattern = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s!()\[\]{};:'".,<>?]))'''
    re_pattern = re.compile(pattern)

    sentences = []

    print("remove links from data")
    data = [re.sub(re_pattern, "", tweet) for tweet in data]

    print("tokenizes tweets")
    for tweet in data:
        tokens = nltk.word_tokenize(tweet)
        tokens = [word for word in tokens if len(word) > 1]
        sentences.append(tokens)
    return sentences


def preprocess_data_user_history(data, stop_at=500):
    """
        Preprocesses the raw data by removing links, tokenizing and ommiting tokens containing of only one character (punctuation, @, #, ...).
        Tweets of users are grouped together to form bigger document in comparison to only tweets.

    :param data: grouped pandas dataframe by userid
    :return: list of token lists
    """

    import re
    pattern = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s!()\[\]{};:'".,<>?]))'''
    re_pattern = re.compile(pattern)

    sentences = []
    print(np.dtype(data["tweet"]))

    print("tokenizes tweets")
    i = 0
    for name, group in data:
        if stop_at != None and i >= stop_at:
            break
        if i%100 == 0:
            print(i)
        i+=1
        user_tweets = group.tweet.values
        one_sentence = ' '.join(sent for sent in user_tweets)
        one_sentence_no_links = re.sub(re_pattern, "", str(one_sentence))
        one_sentence_no_links = one_sentence_no_links.decode('utf-8')
        tokens = nltk.word_tokenize(one_sentence_no_links)
        tokens = [word for word in tokens if len(word) > 1]
        sentences.append(tokens)
    return sentences

def clean_sentences_user_history(data):

    docs_sentences = []
    count = 0
    for name, group in data:
        if count%1000==0:
            print("Processed "+str(count) + "users ...")
        user_tweets = group.tweet_no_links.values
        one_sentence = ' '.join(str(sent) for sent in user_tweets)
        one_sentence = one_sentence.decode('utf-8')
        cleaned_sentences_per_user = sentence_without_stopwords(one_sentence)
        split_cleaned_sentence = cleaned_sentences_per_user.split(" ")
        docs_sentences.append(split_cleaned_sentence)
        count += 1
    return docs_sentences

def remove_stopwords(tokens, language="english"):
    """
    removes stopwords from a list of tokens in a give language.

    :param tokens: list of tokens representing a sentence
    :param language: the language to map stopwords onto

    :return: list of tokens without stopwords
    """

    from nltk.corpus import stopwords
    stopword_list = stopwords.words(language)
    filtered_words = [word for word in tokens if word not in stopword_list]
    return filtered_words


def lemmatize(sentences):
    import spacy
    nlp = spacy.load("en")
    for sent in sentences:
        doc = nlp(sent)
        for token in doc:
            print(token, token.lemma, token.lemma_)


def tokenizeText(sample):
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    from nltk.corpus import stopwords
    import string
    from spacy.en import English
    parser = English()

    # A custom stoplist
    STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
    # List of symbols we don't care about
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "``", "''", "'ve", "..", "--"]

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # numbers
    tokens = [tok for tok in tokens if not tok.isnumeric()]

    # keep tokens with at least three letters
    tokens = [tok for tok in tokens if len(tok) > 2]


    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens


def remove_links(text):
    import re
    try:
        pattern = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s!()\[\]{};:'".,<>?]))'''
        re_pattern = re.compile(pattern)
        sentence_no_links = re.sub(re_pattern, "", text)
        return sentence_no_links
    except:
        return text


def preprocess_spacy_tokenize(data):
    import lda2vec.preprocess as spacy_preprocess

    clean_sentences = sentences_without_stopwords(data)

    tokens, vocab = spacy_preprocess.tokenize(clean_sentences, max_length=10000, merge=False, n_threads=4)
    return tokens, vocab


def sentences_without_stopwords(data, do_print=False):
    clean_sentences = []
    for sent in data:
        tokens = tokenizeText(sent)
        text = ' '.join(p for p in tokens)
        if do_print:
            print(text)
        clean_sentences.append(text)
    return clean_sentences

def sentence_without_stopwords(sentence):
    tokens = tokenizeText(sentence)
    text = ' '.join(p for p in tokens)
    return text