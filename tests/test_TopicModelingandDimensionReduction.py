from unittest import TestCase
from gensim import corpora, models
import numpy as np
import TopicModelingandDimensionReduction as tm


# Define corpus of documents
corpus = [['apple', 'banana', 'orange', 'peach'], ['banana', 'orange', 'peach', 'grape'],
                  ['orange', 'peach', 'grape', 'kiwi'], ['peach', 'grape', 'kiwi', 'apple'],
                  ['grape', 'kiwi', 'apple', 'banana']]
# Create dictionary from corpus
dictionary = corpora.Dictionary(corpus)
# Create bag-of-words representation of corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]

# counts of frequency
matrix = np.array([[1, 2, 0], [2, 1, 0], [0, 0, 2], [0, 0, 2]])

document_fruit = ["apple banana grape orange", "peach orange kiwi apple", "grape banana peach apple"]

class Test(TestCase):
    def test_fit_lda(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics)
        # Print topics and their top words
        for topic in lda_model.print_topics():
            print(topic)

    def test_fit_lda_matrixinput(self):
        num_topics = 2

        lda_model = tm.fit_lda(counts=matrix, numTopics=num_topics)
        # Print topics and their top words
        for topic in lda_model.print_topics():
            print(topic)

    def test_fit_lsa(self):
        numcomponents = 2
        lsa_model = tm.fit_lsa(bag=bow_corpus, numComponents=numcomponents)
        for topic in lsa_model.print_topics():
            print(topic)

    def test_fit_lsa_matrixinput(self):
        numcomponents = 2
        lsa_model = tm.fit_lsa(counts=matrix, numComponents=numcomponents)
        for topic in lsa_model.print_topics():
            print(topic)

    def test_resume_lda_model(self):
        num_topics = 4
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics, counts=None, Solver='cvb0')
        tm.resume_lda_model(ldamdl=lda_model, bag=bow_corpus)
        for topic in lda_model.print_topics():
            print(topic)

    def test_resume_lda_model_matrixinput(self):
        num_topics = 4
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics, counts=None, Solver='cvb0')
        tm.resume_lda_model(ldamdl=lda_model,counts=matrix)
        for topic in lda_model.print_topics():
            print(topic)

    def test_logprob_and_gdns_lda(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics)
        logprob, ppl = tm.logprob_and_gdns_lda(ldamdl=lda_model, bag=bow_corpus)
        print(logprob)
        print(ppl)

    def test_logprob_and_gdns_lda_document_input(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics)
        logprob, ppl= tm.logprob_and_gdns_lda(ldamdl=lda_model, documents=document_fruit)
        print(logprob)
        print(ppl)

    def test_logprob_and_gdns_lda_matrix_input(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics)
        logprob, ppl = tm.logprob_and_gdns_lda(ldamdl=lda_model,counts=matrix)
        print(logprob)
        print(ppl)

    def test_predict(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics)
        topicIdx, score= tm.predict(bag=bow_corpus, ldamdl=lda_model)
        print(topicIdx)
        print(score)

    def test_predict_document_input(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics)
        topicIdx, score= tm.predict(documents=document_fruit, ldamdl=lda_model)
        print(topicIdx)
        print(score)

    def test_predict_matrix_input(self):
        num_topics = 2
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics)
        topicIdx, score= tm.predict(counts=matrix, ldamdl=lda_model)
        print(topicIdx)
        print(score)


    def test_transform_to_lowdimension(self):
        num_topics = 4
        lda_model = tm.fit_lda(bag=bow_corpus, numTopics=num_topics, counts=None)

        dscore_lda_documents = tm.transform_to_lowdimension(ldamdl=lda_model, documents=document_fruit)
        dscore_lda_bag = tm.transform_to_lowdimension(ldamdl=lda_model, bag=bow_corpus)
        dscore_lda_count = tm.transform_to_lowdimension(ldamdl=lda_model, counts=matrix)

        print(dscore_lda_documents)
        print(dscore_lda_bag)
        print(dscore_lda_count)

        lsa_model = tm.fit_lsa(bag=bow_corpus, numComponents=num_topics)

        dscore_lsa_count = tm.transform_to_lowdimension(lsamdl=lsa_model, counts=matrix)
        dscore_lsa_bag = tm.transform_to_lowdimension(lsamdl=lsa_model, bag=bow_corpus)
        dscore_lsa_documents = tm.transform_to_lowdimension(lsamdl=lsa_model, documents=document_fruit)

        print(dscore_lsa_count)
        print(dscore_lsa_bag)
        print(dscore_lsa_documents)