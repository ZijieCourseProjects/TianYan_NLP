def fit_lda(bag, numTopics, counts,
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

    raise NotImplementedError


def fit_lsa(bag, numTopics, counts, DocumentsIn='rows', FeatureStrengthExponent=2):
    """

    :param bag: Input bag-of-words or bag-of-n-grams model
    :param numTopics: Number of topics e.g. 200
    :param counts: Frequency counts of words, specified as a matrix of nonnegative integers
    :param DocumentsIn:Orientation of documents in the word count matrix  values: 'rows' (default) | 'columns'
    :param FeatureStrengthExponent:Initial feature strength exponent value:2(default) | nonnegative scalar
    :return:Output LSA model, returned as an lsaModel object.
    """

    raise NotImplementedError


def resume_lda_model(ldamdl, bag, counts,
                     LogLikelihoodTolerance=0.0001, FitTopicProbabilities=None, FitTopicConcentration=None, DocumentsIn='rows',
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

    raise NotImplementedError


def logprob_and_gdns_lda(ldamdl, documents, bag, counts, DocumentsIn='rows', NumSamples = 1000):
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

    raise NotImplementedError


def predict(ldamdl, documents, bag, counts, DocumentsIn='rows', IterationLimit=100, LogLikelihoodTolerance=0.0001):
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

    raise NotImplementedError


def transform_to_lowdimension(lsamdl, ldamdl, documents, bag, counts,
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

    raise NotImplementedError