import gensim.models
from gensim import corpora, models
from sklearn.decomposition import TruncatedSVD
from gensim.matutils import Sparse2Corpus
from scipy.sparse import csr_matrix
import numpy as np

def fit_lda(numTopics, counts=None, bag=None,
            Solver='cgs', LogLikelihoodTolerance=0.0001, FitTopicProbabilities=True, FitTopicConcentration=None,
            InitialTopicConcentration=4, TopicOrder='initial-fit-probability', WordConcentration=1, DocumentsIn='rows',
            IterationLimit=100,
            DataPassLimit=1, MiniBatchLimit=None, MiniBatchSize=1000, LearnRateDecay=0.5,
            ValidationData=None, ValidationFrequency=None, Verbose=1):
    """
    Fit latent Dirichlet allocation (LDA) model

    :param bag: Input bag-of-words or bag-of-n-grams model
    :param numTopics: Number of topics e.g. 200
    :param counts: Frequency counts of words, specified as a matrix of nonnegative integers

    Solver Options

    :param Solver: Solver for optimization  values: 'cgs' (default) | 'savb' | 'avb' | 'cvb0'
    :param LogLikelihoodTolerance:Relative tolerance on log-likelihood
    :param FitTopicProbabilities:Option for fitting topic concentration values: ture | false
    :param FitTopicConcentration:Option for fitting topic concentration values: ture | false
    :param InitialTopicConcentration:Initial estimate of the topic concentration e.g. 25
    :param TopicOrder:Topic order  values: 'initial-fit-probability' (default) | 'unordered'
    :param WordConcentration:Word concentration e.g. 1
    :param DocumentsIn:Orientation of documents in the word count matrix  values: 'rows' (default) | 'columns'

    Batch Solver Options

    :param IterationLimit:Maximum number of iterations e.g. 200

    Stochastic Solver Options

    :param DataPassLimit:Maximum number of passes through the data , supports only the stochastic ('savb') solver. e.g. 2
    :param MiniBatchLimit:Maximum number of mini-batch passes , supports only the stochastic ('savb') solver. e.g. 200
    :param MiniBatchSize:Mini-batch size , supports only the stochastic ('savb') solver. e.g. 512
    :param LearnRateDecay:Learning rate decay , supports only the stochastic ('savb') solver. e.g. 0.5

    Display Options

    :param ValidationData:Validation data to monitor optimization convergence values:[] (default) | bagOfWords object | bagOfNgrams object | sparse matrix of word counts
    :param ValidationFrequency:Frequency of model validation in number of iterations , positive integer
    :param Verbose:Verbosity level , values: 1 | 0
    :return:Output LDA model, returned as an ldaModel object.
    """
    if counts is None and bag is None:
        raise TypeError('no corpus is given!')
    elif counts is not None and bag is not None:
        raise TypeError('two corpus are given!')
    # input is bagofwords
    elif counts is None:
        if TopicOrder == 'initial-fit-probability':
            lda = gensim.models.LdaModel(corpus=bag, num_topics=numTopics, decay=LearnRateDecay, per_word_topics=True)
        else:
            lda = gensim.models.LdaModel(corpus=bag, num_topics=numTopics, decay=LearnRateDecay)
    # input is matrix of word frequncy
    elif bag is None:
        matrix = counts
        # turn matrix into bow
        corpus_of_matrix = [list(zip(range(len(row)), row)) for row in matrix]
        # get vocabulary
        id2word = corpora.Dictionary.from_corpus(corpus_of_matrix)

        if TopicOrder == 'initial-fit-probability':
            lda = gensim.models.LdaModel(corpus=corpus_of_matrix, id2word=id2word, num_topics=numTopics,
                                         decay=LearnRateDecay, per_word_topics=True)
        else:
            lda = gensim.models.LdaModel(corpus=corpus_of_matrix, id2word=corpus_of_matrix, num_topics=numTopics,
                                         decay=LearnRateDecay)

    return lda


def fit_lsa(numComponents, bag=None, counts=None, DocumentsIn='rows', FeatureStrengthExponent=2):
    """

    :param bag: Input bag-of-words or bag-of-n-grams model
    :param numComponents: Number of components e.g. 200
    :param counts: Frequency counts of words, specified as a matrix of nonnegative integers
    :param DocumentsIn:Orientation of documents in the word count matrix  values: 'rows' (default) | 'columns'
    :param FeatureStrengthExponent:Initial feature strength exponent value:2(default) | nonnegative scalar
    :return:Output LSA model, returned as an lsaModel object.
    """
    if counts is None and bag is None:
        raise TypeError('no corpus is given!')
    elif counts is not None and bag is not None:
        raise TypeError('two corpus are given!')

    # input is bagofwords
    elif counts is None:
        lsa = models.LsiModel(corpus=bag,num_topics=numComponents)

    # input is matrix of word frequncy
    elif bag is None:
        matrix = counts
        matrix_sparse = csr_matrix(matrix)
        # 将稀疏矩阵转换为词袋模型
        corpus = Sparse2Corpus(matrix_sparse.transpose())

        lsa = models.LsiModel(corpus=corpus, num_topics=2)

    return lsa


def resume_lda_model(ldamdl:gensim.models.LdaModel, bag=None, counts=None,
                     LogLikelihoodTolerance=0.0001, FitTopicProbabilities=None, FitTopicConcentration=None,
                     DocumentsIn='rows',
                     IterationLimit=100,
                     DataPassLimit=1, MiniBatchLimit=None, MiniBatchSize=1000,
                     ValidationData=None, ValidationFrequency=None, Verbose=1):
    """

    :param ldamdl: Input LDA model
    :param bag: Input bag-of-words or bag-of-n-grams model
    :param counts: Frequency counts of words, specified as a matrix of nonnegative integers

    Solver Options

    :param LogLikelihoodTolerance:Relative tolerance on log-likelihood
    :param FitTopicProbabilities:Option for fitting topic concentration values: ture | false
    :param FitTopicConcentration:Option for fitting topic concentration values: ture | false
    :param DocumentsIn:Orientation of documents in the word count matrix  values: 'rows' (default) | 'columns'

    Batch Solver Options

    :param IterationLimit: Maximum number of iterations e.g. 200

    Stochastic Solver Options

    :param DataPassLimit:Maximum number of passes through the data , supports only the stochastic ('savb') solver. e.g. 2
    :param MiniBatchLimit:Maximum number of mini-batch passes , supports only the stochastic ('savb') solver. e.g. 200
    :param MiniBatchSize:Mini-batch size , supports only the stochastic ('savb') solver. e.g. 512

    Display Options

    :param ValidationData:Validation data to monitor optimization convergence values:[] (default) | bagOfWords object | bagOfNgrams object | sparse matrix of word counts
    :param ValidationFrequency:Frequency of model validation in number of iterations , positive integer
    :param Verbose:Verbosity level , values: 1 | 0
    :return:Output LDA model, returned as an ldaModel object.
    """

    if counts is None and bag is None:
        raise TypeError('no corpus is given!')
    elif counts is not None and bag is not None:
        raise TypeError('two corpus are given!')
    # input is bagofwords
    elif counts is None:
        ldamdl.update(corpus=bag)
    # input is matrix of word frequncy
    elif bag is None:
        matrix = counts
        # turn matrix into bow
        corpus_of_matrix = [list(zip(range(len(row)), row)) for row in matrix]
        # get vocabulary
        ldamdl.update(corpus=corpus_of_matrix)

    return ldamdl


def logprob_and_gdns_lda(ldamdl:gensim.models.LdaModel, documents=None, bag=None, counts=None, DocumentsIn='rows', NumSamples=1000):
    """

    :param ldamdl: Input LDA model
    :param documents: Input documents , valueType: tokenizedDocument array | string array of words | cell array of character vectors
    :param bag: Input bag-of-words or bag-of-n-grams model
    :param counts: Frequency counts of words, specified as a matrix of nonnegative integers
    :param DocumentsIn: Orientation of documents in the word count matrix  values: 'rows' (default) | 'columns'
    :param NumSamples: Number of samples to draw for each document e.g. 500 positive integer

    :return: logProb: Log-probabilities of the documents under the LDA model , numeric vector.
             ppl: Perplexity of the documents calculated from the log-probabilities , positive scalar
    """
    log_probabilitys = []
    if bag is not None:
        for doc_id in range(0, len(bag)):
            doc_bow = bag[doc_id]
            doc_topics = ldamdl.get_document_topics(doc_bow, minimum_probability=0)
            doc_logprob = np.log([p for _, p in doc_topics])
            log_probabilitys.append(doc_logprob)

        ppl = -ldamdl.log_perplexity(chunk=bag)

        return log_probabilitys, ppl

    elif documents is not None:
        dictionary = corpora.Dictionary()
        for text in documents:
            dictionary.add_documents([text.lower().split()])

        corpus_bow = [dictionary.doc2bow(text.lower().split()) for text in documents]

        for doc_id in range(0, len(corpus_bow)):
            doc_bow = corpus_bow[doc_id]
            doc_topics = ldamdl.get_document_topics(doc_bow, minimum_probability=0)
            doc_logprob = np.log([p for _, p in doc_topics])
            log_probabilitys.append(doc_logprob)

        ppl = -ldamdl.log_perplexity(chunk=corpus_bow)
        return log_probabilitys, ppl

    elif counts is not None:
        matrix = counts
        matrix_sparse = csr_matrix(matrix)
        # 将稀疏矩阵转换为词袋模型
        corpus = Sparse2Corpus(matrix_sparse.transpose())
        for doc_id in range(0, len(corpus)):
            doc_bow = corpus[doc_id]
            doc_topics = ldamdl.get_document_topics(doc_bow, minimum_probability=0)
            doc_logprob = np.log([p for _, p in doc_topics])
            log_probabilitys.append(doc_logprob)

        ppl = -ldamdl.log_perplexity(chunk=corpus)
        return log_probabilitys, ppl

    else:
        print("Please give the a corpus:document | bag-of-word | frequency matrix of word")


def predict(ldamdl, documents=None, bag=None, counts=None, DocumentsIn='rows', IterationLimit=100, LogLikelihoodTolerance=0.0001):
    """

    :param ldamdl: Input LDA model
    :param documents: Input documents , valueType: tokenizedDocument array | string array of words | cell array of character vectors
    :param bag: Input bag-of-words or bag-of-n-grams model
    :param counts: Frequency counts of words, specified as a matrix of nonnegative integers

    :param DocumentsIn: Orientation of documents in the word count matrix  values: 'rows' (default) | 'columns'
    :param IterationLimit: Maximum number of iterations e.g. 200
    :param LogLikelihoodTolerance: Relative tolerance on log-likelihood e.g. 0.001 | positive scalar

    :return: topicIdx: Predicted topic indices , vector of numeric indices.
             score: Predicted topic probabilities , matrix
    """
    # 每篇文档最可能的主题的索引向量
    top_topics_predict = []
    # 主题出现在文档中的概率矩阵
    topic_scores = []
    if bag is not None:
        for doc_id in range(0, len(bag)):
            doc_bow = bag[doc_id]
            doc_topics = ldamdl.get_document_topics(doc_bow, minimum_probability=0)
            # 将得分写入列表
            doc_score = []
            for tu in doc_topics:
                doc_score.append(tu[1])
            topic_scores.append(doc_score)
            # 计算得分最高的主题
            top_topic = 0
            probability = doc_topics[0][1]
            for i in range(1, len(doc_topics)):
                if doc_topics[i][1] > probability:
                    top_topic = i
                    probability = doc_topics[i][1]

            top_topics_predict.append(top_topic)

        topic_scores = np.array(topic_scores)
        return top_topics_predict, topic_scores

    elif documents is not None:
        dictionary = corpora.Dictionary()
        for text in documents:
            dictionary.add_documents([text.lower().split()])

        corpus_bow = [dictionary.doc2bow(text.lower().split()) for text in documents]

        for doc_id in range(0, len(corpus_bow)):
            doc_bow = corpus_bow[doc_id]
            doc_topics = ldamdl.get_document_topics(doc_bow, minimum_probability=0)
            # 将得分写入列表
            doc_score = []
            for tu in doc_topics:
                doc_score.append(tu[1])
            topic_scores.append(doc_score)
            # 计算得分最高的主题
            top_topic = 0
            probability = doc_topics[0][1]
            for i in range(1, len(doc_topics)):
                if doc_topics[i][1] > probability:
                    top_topic = i
                    probability = doc_topics[i][1]

            top_topics_predict.append(top_topic)

        topic_scores = np.array(topic_scores)
        return top_topics_predict, topic_scores

    elif counts is not None:
        matrix = counts
        matrix_sparse = csr_matrix(matrix)
        # 将稀疏矩阵转换为词袋模型
        corpus = Sparse2Corpus(matrix_sparse.transpose())
        for doc_id in range(0, len(corpus)):
            doc_bow = corpus[doc_id]
            doc_topics = ldamdl.get_document_topics(doc_bow, minimum_probability=0)
            # 将得分写入列表
            doc_score = []
            for tu in doc_topics:
                doc_score.append(tu[1])
            topic_scores.append(doc_score)
            # 计算得分最高的主题
            top_topic = 0
            probability = doc_topics[0][1]
            for i in range(1, len(doc_topics)):
                if doc_topics[i][1] > probability:
                    top_topic = i
                    probability = doc_topics[i][1]

            top_topics_predict.append(top_topic)

        topic_scores = np.array(topic_scores)
        return top_topics_predict, topic_scores

    else:
        print("Please give the a corpus:document | bag-of-word | frequency matrix of word")


def transform_to_lowdimension(lsamdl=None, ldamdl=None, documents=None, bag=None, counts=None,
                              DocumentsIn='rows', IterationLimit=100, LogLikelihoodTolerance=0.0001):
    """

    :param lsamdl: Input LSA model
    :param ldamdl: Input LDA model
    :param documents: Input documents , valueType: tokenizedDocument array | string array of words | cell array of character vectors
    :param bag: Input bag-of-words or bag-of-n-grams model
    :param counts: Frequency counts of words, specified as a matrix of nonnegative integers

    :param DocumentsIn: Orientation of documents in the word count matrix  values: 'rows' (default) | 'columns'
    :param IterationLimit: Maximum number of iterations e.g. 200
    :param LogLikelihoodTolerance: Relative tolerance on log-likelihood e.g. 0.001 | positive scalar

    :return: dscores: Output document scores , matrix of score vectors.
    """

    dscore_matrix = []
    # use lda model to transform
    if ldamdl is not None:
        # documents input
        if documents is not None:
            new_word_list = [doc.split() for doc in documents]
            dictionary = corpora.Dictionary(new_word_list)
            new_corpus = [dictionary.doc2bow(doc.split()) for doc in documents]

            new_doc_topic_dist = ldamdl[new_corpus]

            # 打印转换后的主题分布向量矩阵
            for doc in new_doc_topic_dist:
                dscore_of_doc = [tu[1] for tu in doc[0]]

                dscore_matrix.append(dscore_of_doc)

            dscore_matrix = np.array(dscore_matrix)

            return dscore_matrix
        # bag input
        if bag is not None:
            new_bag_topic_dist = ldamdl[bag]

            for doc in new_bag_topic_dist:
                dscore_of_doc = [tu[1] for tu in doc[0]]

                dscore_matrix.append(dscore_of_doc)

            dscore_matrix = np.array(dscore_matrix)

            return dscore_matrix

        # matrix input
        if counts is not None:
            matrix = counts
            matrix_sparse = csr_matrix(matrix)
            # 将稀疏矩阵转换为词袋模型
            corpus = Sparse2Corpus(matrix_sparse.transpose())

            new_count_topic_dist = ldamdl[corpus]

            for doc in new_count_topic_dist:
                dscore_of_doc = [tu[1] for tu in doc[0]]

                dscore_matrix.append(dscore_of_doc)

            dscore_matrix = np.array(dscore_matrix)

            return dscore_matrix

        else:
            print("Please give the a corpus:document | bag-of-word | frequency matrix of word")

    # use lsa model to transform
    if lsamdl is not None:
        # documents input
        if documents is not None:
            new_word_list = [doc.split() for doc in documents]
            dictionary = corpora.Dictionary(new_word_list)
            new_corpus = [dictionary.doc2bow(doc.split()) for doc in documents]

            vectors = lsamdl[new_corpus]

            dscore_matrix = np.zeros((len(new_corpus), lsamdl.num_topics))
            i = 0
            for vec in vectors:
                for tu in vec:
                    dscore_matrix[i, tu[0]] = tu[1]
                i = i + 1

            return dscore_matrix
        # bag input
        if bag is not None:
            vectors = lsamdl[bag]

            dscore_matrix = np.zeros((len(bag), lsamdl.num_topics))
            i = 0
            for vec in vectors:
                for tu in vec:
                    dscore_matrix[i, tu[0]] = tu[1]
                i = i + 1

            return dscore_matrix

        # matrix input
        if counts is not None:
            matrix = counts
            matrix_sparse = csr_matrix(matrix)
            # 将稀疏矩阵转换为词袋模型
            corpus = Sparse2Corpus(matrix_sparse.transpose())
            vectors = lsamdl[corpus]

            dscore_matrix = np.zeros((matrix.shape[0], lsamdl.num_topics))
            i = 0
            for vec in vectors:
                for tu in vec:
                    dscore_matrix[i, tu[0]] = tu[1]
                i = i + 1

            return dscore_matrix

        else:
            print("Please give the a corpus:document | bag-of-word | frequency matrix of word")



