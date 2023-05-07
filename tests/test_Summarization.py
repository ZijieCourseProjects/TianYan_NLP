from unittest import TestCase
from Summarization import *
from Document import tokenize_text

class Test(TestCase):
    def test_extractSummarya(self):
        str = '''
            The quick brown fox jumped over the lazy dog.
            The fox jumped over the dog.
            The lazy dog saw a fox jumping.
            There seem to be animals jumping other animals.
            There are quick animals and lazy animals.
            '''

        test_ans=('The lazy dog saw a fox jumping. The quick brown fox jumped over the lazy dog. \
        There seem to be animals jumping other animals. There are quick animals and lazy animals. \
        The fox jumped over the dog.', [0.8575369715690613, 0.8529646694660187, 0.7511043846607208, \
                                        0.7467152625322342, 0.6937024891376495])
        documents = tokenize_text(str, 1)
        tb=extractSummary(documents)
        self.assertEqual(tb, test_ans)

    def test_rakeKeywords(self):
        from Summarization import rakeKeywords
        textData = '''
            MATLAB provides tools for scientists and engineers. MATLAB is used by scientists and engineers.
            Analyze text and images. You can import text and images.
            Analyze text and images. Analyze text, images, and videos in MATLAB.
            '''

        test_ans={'matlab provides tools': 7.666666666666667, 'import text': 4.0, 'analyze text': 4.0, \
                  'matlab': 1.6666666666666667, 'videos': 1.0, 'used': 1.0, \
                  'scientists': 1.0, 'images': 1.0, 'engineers': 1.0}
        documents = tokenize_text(textData, 1)
        tb1= rakeKeywords(documents)
        self.assertEqual(tb1, test_ans)

    def test_textrankKeyword(self):
        from Summarization import textrankKeyword

        textData = '''
            MATLAB provides really useful tools for engineers. Scientists use many useful tools in MATLAB.
            MATLAB and Simulink have many features. Use MATLAB and Simulink for engineering workflows.
            Analyze text and images in MATLAB. Analyze text, images, and videos in MATLAB.
            '''

        test_ans={'MATLAB': 0.2078737777433317, 'many useful tools': 0.2078240153351183, \
                  'many features': 0.18454041374603997, 'engineering workflows': 0.17369058539076326, \
                  'Simulink': 0.13967105513518668, 'engineers': 0.12060854761404856, \
                  'Analyze text': 0.10020820248342284, 'really useful tools': 0.09727274108283426, \
                  'videos': 0.09559604727124925, 'images': 0.09223987387898983, 'text': 0.07968285656435065, \
                  'Scientists': 0.07033872827485316}
        documents = tokenize_text(textData, 1)
        tb1 = textrankKeyword(documents)
        self.assertEqual(tb1, test_ans)

    def test_bleuEvaluationScore(self):
        from Document import tokenize_text
        from Summarization import bleuEvaluationScore

        str = '''
            The fox jumped over the dog.
            The fast brown fox jumped over the lazy dog.
            The lazy dog saw a fox jumping.
            There seem to be animals jumping other animals.
            There are quick animals and lazy animals
            '''
        d = tokenize_text(str, 1)
        str = '''
            The quick brown animal jumped over the lazy dog.
            The quick brown fox jumped over the lazy dog.
            '''
        test_ans=0.23366176703090105
        r = tokenize_text(str, 1)
        score = bleuEvaluationScore(d.tokens(), r.tokens())
        self.assertEqual(score, test_ans)

    def test_rougeEvaluationScore(self):
        from Summarization import rougeEvaluationScore
        str1 = "the fast brown fox jumped over the lazy dog";
        str2 = '''
            the quick brown animal jumped over the lazy dog,
            the quick brown fox jumped over the lazy dog
            '''
        test_ans=0.8888888888888888
        score = rougeEvaluationScore(str1, str2)
        self.assertEqual(score['rouge1'][0], test_ans)

    def test_bm25Similarity(self):
        from Summarization import bm25Similarity
        textData = [
            "the quick brown fox jumped over the lazy dog",
            "the fast brown fox jumped over the lazy dog",
            "the lazy dog sat there and did nothing",
            "the other animals sat there watching"];

        test_ans=[[0.9597224955842112, 0.1371032136548873, 0.1371032136548873, 0], \
                  [0.1371032136548873, 0.9597224955842112, 0.1371032136548873, 0], \
                  [0.1371032136548873, 0.1371032136548873, 2.604961059442859, 0.0], \
                  [0, 0, 0.0, 2.793289649628144]]

        similarities = bm25Similarity(textData)
        self.assertEqual(similarities, test_ans)

        textData = [
            "the quick brown fox jumped over the lazy dog",
            "the fast brown fox jumped over the lazy dog",
            "the lazy dog sat there and did nothing",
            "the other animals sat there watching"];

        str = ["a brown fox leaped over the lazy dog",
               "another fox leaped over the dog"];
        test_ans=[[0.1371032136548873, 0.1371032136548873, 0.1371032136548873, 0.0], \
                  [0.06855160682744366, 0.06855160682744366, 0.06855160682744366, 0.0]]
        similarities = bm25Similarity(textData, str)
        self.assertEqual(similarities, test_ans)


    def test_cosineSimilarity(self):
        from Summarization import cosineSimilarity
        textData = [
            "the quick brown fox jumped over the lazy dog",
            "the fast brown fox jumped over the lazy dog",
            "the lazy dog sat there and did nothing",
            "the other animals sat there watching"]
        test_ans=[[1,0.90909091, 0.42640143 ,0.24618298],\
                  [0.90909091, 1,  0.42640143, 0.24618298],\
                  [0.42640143, 0.42640143, 1,  0.4330127 ],\
                  [0.24618298, 0.24618298, 0.4330127,  1]]
        similarities = cosineSimilarity(textData)
        self.assertEqual(similarities, test_ans)

        textData = [
            "the quick brown fox jumped over the lazy dog",
            "the fast brown fox jumped over the lazy dog",
            "the lazy dog sat there and did nothing",
            "the other animals sat there watching"]

        str = ["a brown fox leaped over the lazy dog",
               "another fox leaped over the dog"]
        test_ans=[[0.79772404, 0.61545745],\
                  [0.79772404, 0.61545745],\
                  [0.40089186, 0.28867513],\
                  [0.15430335, 0.16666667]]

        similarities = cosineSimilarity(textData, str)
        self.assertEqual(similarities, test_ans)


