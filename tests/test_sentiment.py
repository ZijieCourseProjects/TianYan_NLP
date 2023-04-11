from unittest import TestCase


class Test(TestCase):
    def test_sentiment_score_vader(self):
        from sentiment import sentiment_score_vader
        from Document import tokenize_text
        document = tokenize_text("The book was VERY good!!!!")
        compound_scores, positive, negative, neutral = sentiment_score_vader(document)
        self.assertGreater(compound_scores, 0)
