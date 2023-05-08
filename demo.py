import sentiment
from Document import tokenize_text
from name_entity_recognizition import addEntityDetails
import nltk

print("Welcome to the NLP demo with TianYan platform")
print("We are going to analysis a sentence about Dean's Travelling during his vocation.\n")

word_to_analysis = "Dean and his friend Ashley really believes that the scenery in TianJin is so beautiful that He wants to invite all of his friends come and see. Dean really enjoys this travelling, because he ate many delicious food in this travelling. "
print("The sentence we are going to analyze is: " + word_to_analysis + "\n")

print("We want to know the sentiment of the sentence, indicating whether Dean enjoy this travelling or not.")
print("Doing Sentiment Analysis: ")
document = tokenize_text(word_to_analysis)

# print(sentiment.sentiment_score_vader(document))
compound_scores, positive, negative, neutral = sentiment.sentiment_score_vader(document)
print("The compound sensitive score of the sentence computing by vader algorithm is: " + str(compound_scores))

if compound_scores > 0:
    print("This means the sentiment of the sentence is positive, indicating Dean enjoy this travelling.")
elif compound_scores < 0:
    print("This means the sentiment of the sentence is negative, indicating Dean does not enjoy this travelling.")

print("\nWe want to know entities in the sentence, such as person, location, etc.")
print("Doing Name Entity Recognition: ")

with_tag_Tokenized_word = addEntityDetails(nltk.word_tokenize(word_to_analysis))
print("Related Persons:" + str(set([word for word, tag in with_tag_Tokenized_word if tag == 'person'])))
print("Related Places:" + str(set([word for word, tag in with_tag_Tokenized_word if tag == 'location'])))

from Summarization import textrankKeyword
print("\nWe want to extract some keywords of the sentence as Tags before posting it online.")
print("Doing Keyword Extraction with TextRank algorithm: ")
keywords = sorted(list(textrankKeyword(document).items()), key=lambda x: x[1], reverse=True)
print("The Top-3 keywords and their scores of the sentence are: " + str(keywords[0:3]))

print("\nHowever, Colin is a crazy scientist, who is trying to make a robot to replace Dean.")
print("He is trying to train a deep learning model to generate a sentence which is similar to Dean's sentence.")
print("First, he need to embed the sentence into a vector.")

from word_embedding import doc2sequence, fastTextWordEmbedding
print("Doing Word Embedding: ")
model = fastTextWordEmbedding()
sequence = doc2sequence(model, [word_to_analysis])
print("The vector of the sentence is: " + str(sequence))

print("\nThen, he need to find a sentence which is most similar to Dean's sentence.")
print("He has a list of sentences, and he want to find the most similar one.")
print("The candidate sentences are: ")

similar_sentence = [
    "Join us for a quick, fun, and easy way to learn Python programming.",
    "A quick brown fox jumps over the lazy dog.",
    "Tom like to eat pizza.",
    "Colin played Resident Evil 4, where he acts as leon and try to save the president's daughter ashley.",
    "Tom traveled to Tianjin and he really enjoys everything he met there."
]

for sent in similar_sentence:
    print(sent)

from Summarization import cosineSimilarity
print("\nComputing Cosine Similarity: ")
similarity = cosineSimilarity([word_to_analysis], similar_sentence)
print("The similarity score of the sentence and the candidate sentences are: " + str(similarity))
print("The most similar sentence is: " + similar_sentence[similarity.argmax()])



