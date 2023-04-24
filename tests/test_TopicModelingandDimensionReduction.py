from unittest import TestCase
from gensim import corpora, models
import TopicModelingandDimensionReduction as tm


# Define corpus of documents
corpus = [['apple', 'banana', 'orange', 'peach'], ['banana', 'orange', 'peach', 'grape'],
                  ['orange', 'peach', 'grape', 'kiwi'], ['peach', 'grape', 'kiwi', 'apple'],
                  ['grape', 'kiwi', 'apple', 'banana']]
# Create dictionary from corpus
dictionary = corpora.Dictionary(corpus)
# Create bag-of-words representation of corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]


class Test(TestCase):
    def test_fit_lda(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics, counts=None)
        # Print topics and their top words
        for topic in lda_model.print_topics():
            print(topic)

    def test_fit_lsa(self):
        num_topics = 2
        lsa_model = tm.fit_lsa(bag=bow_corpus, numTopics=num_topics, counts=None)

    def test_resume_lda_model(self):
        num_topics = 4
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics, counts=None, Solver='cvb0')
        tm.resume_lda_model(ldamdl=lda_model, bag=bow_corpus, counts=None, LogLikelihoodTolerance=0.00001)

    def test_logprob_and_gdns_lda(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics, counts=None, Verbose=0)
        logprob = tm.logprob_and_gdns_lda(ldamdl=lda_model, documents=None, bag=bow_corpus, counts=None)
        print(logprob)

    def test_predict(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics, counts=None)
        topicIdx = tm.predict(bag=bow_corpus, documents=None, counts=None, ldamdl=lda_model)
        print(topicIdx)

    def test_transform_to_lowdimension(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics, counts=None)
        lsa_model = tm.fit_lsa(bag=bow_corpus, numTopics=num_topics, counts=None)

        dscore_lda = tm.transform_to_lowdimension(ldamdl=lda_model, documents=None, bag=bow_corpus, counts=None)
        dscore_lsa = tm.transform_to_lowdimension(lsamdl=lsa_model, documents=None, bag=bow_corpus, counts=None)

        print(dscore_lda)
        print(dscore_lsa)
