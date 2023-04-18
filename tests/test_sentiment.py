from unittest import TestCase

from Document import tokenize_text
from sentiment import sentiment_score_vader
from sentiment import sentiment_score_ratio


class Test(TestCase):
    def test_sentiment_score_vader_default_scenario(self):
        document = tokenize_text("The book was VERY good!!!!")
        compound_scores, positive, negative, neutral = sentiment_score_vader(document)
        self.assertGreater(compound_scores, 0)
        document = tokenize_text("The book was VERY bad!!!!")
        compound_scores, positive, negative, neutral = sentiment_score_vader(document)
        self.assertLess(compound_scores, 0)

    def test_sentiment_score_with_custom_lexicon(self):
        custom_lexicon = {"bad": 1};
        document = tokenize_text("The book was VERY bad!!!!")
        compound_scores, positive, negative, neutral = sentiment_score_vader(document, lexicon=custom_lexicon)
        self.assertGreater(compound_scores, 0)

    def test_sentiment_score_with_custom_booster(self):
        custom_booster = ["colin"];
        custom_damper = ["eric"]
        simple_document = tokenize_text("The book was god")
        simple_compound_scores, simple_positive, simple_negative, simple_neutral = sentiment_score_vader(
            simple_document)

        boosted_document = tokenize_text("The book was colin god")
        boosted_compound_scores, boosted_positive, boosted_negative, boosted_neutral = sentiment_score_vader(
            boosted_document, booster=custom_booster)

        damped_document = tokenize_text("The book was eric god")
        damped_compound_scores, damped_positive, damped_negative, damped_neutral = sentiment_score_vader(
            damped_document, dampener=custom_damper)

        self.assertGreater(boosted_compound_scores, simple_compound_scores)
        self.assertLess(damped_compound_scores, simple_compound_scores)

    def test_sentiment_score_with_custom_negation(self):
        custom_negation = ["kkk"]
        simple_document = tokenize_text("The book was god")
        simple_compound_scores, simple_positive, simple_negative, simple_neutral = sentiment_score_vader(
            simple_document)

        negated_document = tokenize_text("The book was kkk god")
        negated_compound_scores, negated_positive, negated_negative, negated_neutral = sentiment_score_vader(
            negated_document, negation=custom_negation)

        self.assertGreater(simple_compound_scores, 0)
        self.assertLess(negated_compound_scores, 0)

    def test_ratio_sentiment_in_default_scenario(self):
        document = tokenize_text("the book was good")
        compound_score, positive_score, negative_score = sentiment_score_ratio(document)
        self.assertEqual(compound_score, 1)

        document = tokenize_text("the book was bad")
        compound_score, positive_score, negative_score = sentiment_score_ratio(document)
        self.assertEqual(compound_score, -1)

        document = tokenize_text("good good bad")
        compound_score, positive_score, negative_score = sentiment_score_ratio(document)
        self.assertEqual(compound_score, 1)

    def test_ratio_sentiment_with_custom_threshold(self):
        document = tokenize_text("good good bad")
        compound_score, positive_score, negative_score = sentiment_score_ratio(document, threshold=10)
        self.assertEqual(compound_score, 0)

    def test_ratio_sentiment_with_custom_positive_words(self):
        document = tokenize_text("colin colin hello")
        compound_score, positive_score, negative_score = sentiment_score_ratio(document,lexicon={"colin": 1})
        self.assertEqual(compound_score, 1)