#! ~/venv_lda2vec/bin/python
# -*- coding: utf-8 -*-

# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import pickle

from lda2vec import preprocess, Corpus
import numpy as np


def list_all_unicode(tweets):
    decoded_texts = []
    for text in tweets:
        sentence = ' '.join(word for word in text)
        if isinstance(sentence, unicode):
            decoded_texts.append(sentence)
        else:
            print("Found non unicode")
            decoded_text = sentence.decode('utf-8')
            decoded_texts.append(decoded_text)
    return decoded_texts

def process_data (tokens, vocab, model):
    """
    preprocessing of the data by counting word occurrences and filtering according to these.
    The most frequent words are subsampled, and cleans the vocabulary words according to the
    word2vec models vocabulary

    :param tokens: spacy tokens
    :param vocab: spacy vocabulary
    :param model: word2vec model name
    :return:
    """
    corpus = Corpus()
    # Make a ranked list of rare vs frequent words
    corpus.update_word_count(tokens)
    corpus.finalize()
    # The tokenization uses spaCy indices, and so may have gaps
    # between indices for words that aren't present in our dataset.
    # This builds a new compact index
    compact = corpus.to_compact(tokens)
    # Remove extremely rare words
    pruned = corpus.filter_count(compact, min_count=15)
    # Convert the compactified arrays into bag of words arrays
    bow = corpus.compact_to_bow(pruned)
    # Words tend to have power law frequency, so selectively
    # downsample the most prevalent words
    clean = corpus.subsample_frequent(pruned)
    # Now flatten a 2D array of document per row and word position
    # per column to a 1D array of words. This will also remove skips
    # and OoV words
    doc_ids = np.arange(pruned.shape[0])
    flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
    assert flattened.min() >= 0
    # Fill in the pretrained word vectors
    #n_dim = 300
    fn_wordvc = model
    print("starts to compact word vectors")
    vectors, s, f = corpus.compact_word_vectors(vocab, filename=fn_wordvc)
    print("done with compact word vectors")
    # Save all of the preprocessed files
    print("now saving files")
    pickle.dump(vocab, open('vocab_'+id+'.pkl', 'w'))
    pickle.dump(corpus, open('corpus_'+id+'.pkl', 'w'))
    np.save('flattened_'+id, flattened)
    np.save('doc_ids_'+id, doc_ids)
    np.save('pruned_'+id, pruned)
    #np.save('bow_'+id, bow) Does not seem to be neccessary for lda2vec_run.py
    np.save('vectors_'+id, vectors)




id = "honeypot_clean_revised"

model="honeypot_clean_model_revised"

#tweets_fn = "../lda2vec/tweets_shuffled_no_links.txt"

#with open("sentences_no_stopwords.txt", "rb") as fp:   # Unpickling
#    sentences = pickle.load(fp)

sentences_part_1 = np.load("sentences_no_stopwords_revised_1.npy")
sentences_part_2 = np.load("sentences_no_stopwords_revised_2.npy")
sentences = np.append(sentences_part_1,sentences_part_2)
#sentences = np.concatenate(sentences_part_1, sentences_part_2)
assert sentences[len(sentences_part_1)] == sentences_part_2[0]

sentences = list_all_unicode(sentences)
print(sentences[:3])
print(np.shape(sentences))

print("starts to tokenize")
tokens, vocab = preprocess.tokenize(sentences, max_length=10000, merge=False,
                                    n_threads=4)
print("tokenized")
process_data (tokens, vocab, model=model)
print("finished.")