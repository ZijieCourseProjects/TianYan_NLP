import sys
import os
from unittest import TestCase

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from name_entity_recognizition import *


class Test(TestCase):
    def test_predict_person(self):
        m = hmmEntityModel()
        tbl = m.dataset("dataset/train.txt")
        mdl = trainHMMEntityModel(tbl)
        sentence = "John and Mary moved to Germany."
        document = nltk.word_tokenize(sentence)
        a = predict(mdl, document)
        self.assertListEqual(a, [['John', 'B-PER'], ['and', 'O'], ['Mary', 'B-PER'], ['moved', 'O'], ['to', 'O'],
                                 ['Germany', 'B-LOC'], ['.', 'O']])

    def test_predict_organization(self):
        m = hmmEntityModel()
        tbl = m.dataset("dataset/train.txt")
        mdl = trainHMMEntityModel(tbl)
        sentence = "England is out of the European-Union."
        document = nltk.word_tokenize(sentence)
        a = predict(mdl, document)
        self.assertListEqual(a, [['England', 'B-LOC'], ['is', 'O'], ['out', 'O'], ['of', 'O'], ['the', 'O'],
                                 ['European-Union', 'B-ORG'], ['.', 'O']])

    def test_predict_all(self):
        m = hmmEntityModel()
        tbl = m.dataset("dataset/train.txt")
        mdl = trainHMMEntityModel(tbl)
        sentence = "John is a member of BBC in France."
        document = nltk.word_tokenize(sentence)
        a = predict(mdl, document)
        self.assertListEqual(a, [['John', 'B-PER'], ['is', 'O'], ['a', 'O'], ['member', 'O'], ['of', 'O'], ['BBC', 'B-ORG'], ['in', 'O'], ['France', 'B-LOC'], ['.', 'O']])

    def test_addEntityDetails_all(self):
        sentence = "John is a member of BBC in France."
        document = nltk.word_tokenize(sentence)
        a = addEntityDetails(document)
        self.assertListEqual(a,
                             [['John', 'person'], ['is', 'non-entity'], ['a', 'non-entity'], ['member', 'non-entity'],
                              ['of', 'non-entity'], ['BBC', 'organization'], ['in', 'non-entity'],
                              ['France', 'location'], ['.', 'non-entity']])

    def test_addEntityDetails_location(self):
        sentence = "USA is near to Canada."
        document = nltk.word_tokenize(sentence)
        a = addEntityDetails(document)
        self.assertListEqual(a,
                             [['USA', 'location'], ['is', 'non-entity'], ['near', 'non-entity'], ['to', 'non-entity'],
                              ['Canada', 'location'], ['.', 'non-entity']])

    def test_addEntityDetails_default(self):
        sentence = "today is a good day."
        document = nltk.word_tokenize(sentence)
        a = addEntityDetails(document)
        self.assertListEqual(a, [['today', 'non-entity'], ['is', 'non-entity'], ['a', 'non-entity'],
                                 ['good', 'non-entity'], ['day', 'non-entity'], ['.', 'non-entity']])
