from lda2vec import preprocess, Corpus
import numpy as np

# Fetch data
texts = ["Republican frustration with President Trump is boiling over in the wake of his incendiary tweets on Thursday attacking MSNBC host Mika Brzezinski.",
         "After months of winding through the courts, the so-called 'watered-down', revised version of President Donald Trump's fiercely litigated travel ban finally went into effect at 8 p.m. ET Thursday.",
         "Millions of Americans of all ages and needs would be affected if Republicans in Congress succeed in overhauling major parts of the Affordable Care Act. And the latest maneuvering is only intensifying concerns.",
         "BERLIN - Germany on Friday recognized the right of same-sex couples to wed, a major step for gay women and men living in a country split between conservative, Christian customs and modernizing forces."]




# Preprocess data
max_length = 10000   # Limit of 10k words per document
# Convert to unicode (spaCy only works with unicode)
texts = [unicode(d) for d in texts]
tokens, vocab = preprocess.tokenize(texts, max_length, merge=False,
                                    n_threads=4)

print(tokens, vocab)
corpus = Corpus()
# Make a ranked list of rare vs frequent words
corpus.update_word_count(tokens)
corpus.finalize()
compact = corpus.to_compact(tokens)
# Remove extremely rare words
pruned = corpus.filter_count(compact, min_count=30)
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
n_dim = 300
fn_wordvc = 'test.bin'
vectors, s, f = corpus.compact_word_vectors(vocab, filename=fn_wordvc)
print(bow, vectors, doc_ids,flattened)
# Save all of the preprocessed files
"""
pickle.dump(vocab, open('vocab.pkl', 'w'))
pickle.dump(corpus, open('corpus.pkl', 'w'))
np.save("flattened", flattened)
np.save("doc_ids", doc_ids)
np.save("pruned", pruned)
np.save("bow", bow)
np.save("vectors", vectors)"""