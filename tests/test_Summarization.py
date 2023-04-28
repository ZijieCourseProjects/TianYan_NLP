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
        documents = tokenize_text(str, 1)
        extractSummary(documents)

    def test_rakeKeywords(self):
        from Summarization import rakeKeywords
        textData = '''
            MATLAB provides tools for scientists and engineers. MATLAB is used by scientists and engineers.
            Analyze text and images. You can import text and images.
            Analyze text and images. Analyze text, images, and videos in MATLAB.
            '''
        documents = tokenize_text(textData, 1)
        tb1= rakeKeywords(documents)
        print(tb1)

    def test_textrankKeyword(self):
        from Summarization import textrankKeyword

        textData = '''
            MATLAB provides really useful tools for engineers. Scientists use many useful tools in MATLAB.
            MATLAB and Simulink have many features. Use MATLAB and Simulink for engineering workflows.
            Analyze text and images in MATLAB. Analyze text, images, and videos in MATLAB.
            '''
        documents = tokenize_text(textData, 1)
        tb1 = textrankKeyword(documents)
        print(tb1)

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
        r = tokenize_text(str, 1)
        score = bleuEvaluationScore(d.tokens(), r.tokens())
        print(score)

    def test_rougeEvaluationScore(self):
        str1 = "the fast brown fox jumped over the lazy dog";
        str2 = '''
            the quick brown animal jumped over the lazy dog,
            the quick brown fox jumped over the lazy dog
            '''
        score = rougeEvaluationScore(str1, str2)
        print(score)

    def test_bm25Similarity(self):
        from Summarization import bm25Similarity
        textData = [
            "the quick brown fox jumped over the lazy dog",
            "the fast brown fox jumped over the lazy dog",
            "the lazy dog sat there and did nothing",
            "the other animals sat there watching"];

        similarities = bm25Similarity(textData)
        print(similarities)

        textData = [
            "the quick brown fox jumped over the lazy dog",
            "the fast brown fox jumped over the lazy dog",
            "the lazy dog sat there and did nothing",
            "the other animals sat there watching"];

        str = ["a brown fox leaped over the lazy dog",
               "another fox leaped over the dog"];

        similarities = bm25Similarity(textData, str)
        print(similarities)


    def test_cosineSimilarity(self):
        from Summarization import cosineSimilarity
        textData = [
            "the quick brown fox jumped over the lazy dog",
            "the fast brown fox jumped over the lazy dog",
            "the lazy dog sat there and did nothing",
            "the other animals sat there watching"];

        similarities = cosineSimilarity(textData)
        print(similarities)

        textData = [
            "the quick brown fox jumped over the lazy dog",
            "the fast brown fox jumped over the lazy dog",
            "the lazy dog sat there and did nothing",
            "the other animals sat there watching"];

        str = ["a brown fox leaped over the lazy dog",
               "another fox leaped over the dog"];

        similarities = cosineSimilarity(textData, str)
        print(similarities)


