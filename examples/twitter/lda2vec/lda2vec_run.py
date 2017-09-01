# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os
import os.path
import pickle
import time

import chainer
from chainer import cuda
from chainer import serializers
import chainer.optimizers as O
import numpy as np

from lda2vec import utils
from lda2vec import prepare_topics, print_top_words_per_topic, topic_coherence
from lda2vec_model import LDA2Vec


class LDA2VEC_run():

    def __init__(self, id="honeypot_1000", resultmodelid = "honeypot_1000_2", doprint=True, n_topics=5, batchsize=4096, power=0.75, pretrained=True, temperature=1.0, n_units=300):
        """
        Sets Model Parameters

        :param id: Identifier Name for a LDA2VEC RUN
        :param n_topics: Number of topics to fit
        :param batchsize:
        :param power: Power for neg sampling
        :param pretrained: Intialize with pretrained word vectors
        :param temperature: Sampling temperature
        :param n_units: Number of dimensions in a single word vector
        """

        gpu_id = int(os.getenv('CUDA_GPU', 0))
        cuda.get_device(gpu_id).use()
        print "Using GPU " + str(gpu_id)

        self.id = id
        self.modelid = resultmodelid
        self.doprint=doprint
        self.n_topics = n_topics
        self.batchsize = batchsize
        self.power = power
        self.pretrained = pretrained
        self.temperature = temperature
        self.n_units = n_units
        self.gpu_id = 0
        cuda.get_device().use()

        self.vocab = None
        self.corpus = None
        self.flattened = None
        self.doc_ids = None
        self.vectors = None


    def load_preprocessed_files(self):
        """
        loads the preprocessed files created from ../data/preprocess*.py

        :return:
        """
        data_dir = os.getenv('data_dir', '../data/')
        fn_vocab = ('{data_dir:s}/vocab_'+self.id+'.pkl').format(data_dir=data_dir)
        fn_corpus = ('{data_dir:s}/corpus_'+self.id+'.pkl').format(data_dir=data_dir)
        fn_flatnd = ('{data_dir:s}/flattened_'+self.id+'.npy').format(data_dir=data_dir)
        fn_docids = ('{data_dir:s}/doc_ids_'+self.id+'.npy').format(data_dir=data_dir)
        fn_vectors = ('{data_dir:s}/vectors_'+self.id+'.npy').format(data_dir=data_dir)
        self.vocab = pickle.load(open(fn_vocab, 'r'))
        self.corpus = pickle.load(open(fn_corpus, 'r'))
        self.flattened = np.load(fn_flatnd)
        self.doc_ids = np.load(fn_docids)
        self.vectors = np.load(fn_vectors)


    def init_model(self, n_samples=15):
        """
        initializes the LDA2EC model

        :return:
        """

        # number of documents
        n_docs = self.doc_ids.max() + 1

        # Number of unique words in the vocabulary
        self.n_vocab = self.flattened.max() + 1

        # Get the string representation for every compact key
        self.words = self.corpus.word_list(self.vocab)[:self.n_vocab]

        # How many tokens are in each document
        doc_idx, lengths = np.unique(self.doc_ids, return_counts=True)
        self.doc_lengths = np.zeros(self.doc_ids.max() + 1, dtype='int32')
        self.doc_lengths[doc_idx] = lengths

        # Count all token frequencies
        tok_idx, freq = np.unique(self.flattened, return_counts=True)
        self.term_frequency = np.zeros(self.n_vocab, dtype='int32')
        self.term_frequency[tok_idx] = freq

        self.model = LDA2Vec(n_documents=n_docs, n_document_topics=self.n_topics,
                        n_units=self.n_units, n_vocab=self.n_vocab, counts=self.term_frequency,
                        n_samples=n_samples, power=self.power, temperature=self.temperature)


    def start_optimizing(self, threshold=5.0, epochs = 200, do_print=False):
        """
        after initialization of the model, this function initializes the Adam Optimizer with a threshold
        and creates topics for a number of epochs.

        :param threshold:
        :param do_print:
        :return:
        """
        if self.pretrained:
            self.model.sampler.W.data[:, :] = self.vectors[:self.n_vocab, :]
        self.model.to_gpu()
        # Adam optimizer is an extension to stochastic gradient descent for deep learning apps
        # per-parameter learning rates, improves performance on problems with sparse gradients (NLP),
        # adapted based on the averages on recent magnitudes of the gradients for the weights, so the algorithm
        # does well on noisy data
        self.optimizer = O.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
        # setup() prepares for the optimization given a link.
        self.optimizer.setup(self.model)
        clip = chainer.optimizer.GradientClipping(threshold=threshold)
        # hook function for Gradient Clipping is called after the gradient computation and right before the actual update of parameters
        self.optimizer.add_hook(clip)

        self.fraction = self.batchsize * 1.0 / self.flattened.shape[0]
        for epoch in range(epochs):
            print(epoch)
            self.create_topics(do_print, epoch)

    def create_topics(self, do_print, epoch):
        """
        for an epoch this function prepares topics with given corpus word list and
        extracts top words from those.
        It stores intermediate results of the data in pyldavis file and the model in a hdf5 file .

        <<<< This is the LDA part >>>>

        :param do_print: print top words in an epoch
        :param epoch: index of an epoch
        :return:
        """
        j=0
        # prepare the topic_term_distributions, document_topic_distributions and term_frequencies using softmax
        data = prepare_topics(weights=cuda.to_cpu(self.model.mixture.weights.W.data).copy(),
                              topic_vectors=cuda.to_cpu(self.model.mixture.factors.W.data).copy(),
                              word_vectors=cuda.to_cpu(self.model.sampler.W.data).copy(),
                              vocab=self.words, doprint=False)

        #top_words = print_top_words_per_topic(data, do_print=do_print)
        #if j % 100 == 0 and j > 100 and do_print:
        #    coherence = topic_coherence(top_words)
        #    for j in range(self.n_topics):
        #        print j, coherence[(j, 'cv')]
        data['doc_lengths'] = self.doc_lengths
        data['term_frequency'] = self.term_frequency
        np.savez('topics_' + self.modelid + '.pyldavis', **data)
        for d, f in utils.chunks(self.batchsize, self.doc_ids, self.flattened):
            self.update_per_chunk(d, epoch, f)
            j+=1
        # saves the parameters of model into a file in hdf5 format
        serializers.save_hdf5("lda2vec_" + self.modelid + ".hdf5", self.model)

    def update_per_chunk(self, d, epoch, f):
        """
        fits the model to the partional chunks of data d and f, calculates the loss with the model
        prior and fraction and updates the optimizer based on this.

        :param d: doc_ids
        :param epoch: current epoch (mainly for logging)
        :param f: word_indices
        :return:
        """

        t0 = time.time()
        # cleargrads() is introduced in v1.15 to replace zerograds() for efficiency. Gradients always need to be cleared before the update method
        self.optimizer.target.cleargrads()
        # predict word given context and pivot word ~ looks like word2vec here
        l = self.model.fit_partial(d.copy(), f.copy())
        # calculate the log likelihood of the observed topic proportions
        prior = self.model.prior()
        loss = prior * self.fraction
        # loss variable holds the history of computation (or computational graph), which enables us to compute its differentiation using backward()
        # error backpropagation
        # Gradients of parameters are computed by the backward() method. Note that gradients are accumulated by the method
        loss.backward()
        if self.doprint:
            print "Prior: ", prior, " Loss: ", loss
        # after computation of gradients, we update the optimizers parameters of the target link (=model)
        self.optimizer.update()
        msg = ("J:{j:05d} E:{epoch:05d} L:{loss:1.3e} "
               "P:{prior:1.3e} R:{rate:1.3e}")

        prior.to_cpu()
        loss.to_cpu()

        t1 = time.time()
        dt = t1 - t0
        rate = self.batchsize / dt
        if self.doprint:
            logs = dict(loss=float(l), epoch=epoch,
                        prior=float(prior.data), rate=rate)
            print msg.format(**logs)


    def main(self):
        self.load_preprocessed_files()
        self.init_model()
        self.start_optimizing()

lda2vec = LDA2VEC_run(id="honeypot_clean_revised", resultmodelid="honeypot_clean_revised", doprint=False, n_topics=20)
lda2vec.main()
print("finish")