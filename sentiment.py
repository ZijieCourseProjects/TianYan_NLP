import os
import numpy as np
import warnings

from senti.vaderSentiment import SentimentIntensityAnalyzer
import codecs


def sentiment_score_vader(document, lexicon=None, booster=None, dampener=None, negation=None):
    """
    Sentiment analysis using VADER algorithm

    :param document: A document
    :param lexicon: map of words to sentiment scores, e.g. {"bad": -1}
    :param booster: list of words to boost sentiment scores, e.g. ["very", "extremely"]
    :param dampener: list of words to dampen sentiment scores, e.g. ["slightly", "somewhat"]
    :param negation: list of words to negate sentiment scores, e.g. ["not", "never"]
    :return: (compound_scores, positive, negative, neutral)
    """
    analyzer = SentimentIntensityAnalyzer(custom_lexicon=lexicon, custom_negation=negation, custom_booster=booster,
                                          custom_damper=dampener)
    scores = analyzer.polarity_scores(document.original_text())
    return scores["compound"], scores["pos"], scores["neg"], scores["neu"]


def sentiment_score_ratio(document, lexicon=None, threshold=None):
    """
    Ratio of positive and negative sentiment scores

    :param document: A document
    :param lexicon: map of words to sentiment scores, e.g. {"bad": -1}
    :param threshold: If the ratio of the positive score to negative score of documents(i) is larger than Threshold, then compoundScores(i) is 1. If the ratio of the negative score to positive score of documents(i) is larger than Threshold, then compoundScores(i) is -1. Otherwise, compoundScores(i) is 0.
    :return: (compound_scores, positive, negative)
    """
    if lexicon is None:
        lexicon = {}
        with codecs.open(os.path.join(os.getcwd(),"../senti/vader_lexicon.txt"), encoding='utf-8') as f:
            lexicon_full_filepath = f.read()
            for line in lexicon_full_filepath.rstrip('\n').split('\n'):
                if not line:
                    continue
                (word, measure) = line.strip().split('\t')[0:2]
                lexicon[word] = float(measure)

    positive = 0
    negative = 0

    for token in document.tokens():
        if token in lexicon:
            if lexicon[token] > 0:
                positive += lexicon[token]
            else:
                negative += lexicon[token]

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if abs(negative) > abs(positive):
        compound = -np.abs(np.float64(negative) / np.float64(positive))
    elif abs(negative) < abs(positive):
        compound = np.abs(np.float64(positive) / np.float64(negative))

    threshold = 1 if threshold is None else threshold

    if compound > threshold:
        compound = 1
    elif compound < -threshold:
        compound = -1
    else:
        compound = 0

    return compound, positive, negative

