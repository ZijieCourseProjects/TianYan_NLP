import nltk
import re
import string
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize as nlkt_sent_tokenize
from nltk.tokenize import word_tokenize as nlkt_word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
from scipy.spatial.distance import cosine


#删除字符串中'\n'和''
def eliminata_list_token(references:list):
    for i in range(len(references) - 1, -1,-1):  # 同样不能用正序循环，for i in range(0,len(alist)), 用了remove()之后，len(alist)是动态的，会产生列表下标越界错误
        if references[i] == '\n':
            references.remove('\n')  # 从左往右删除首次出现的值为‘d'的元素

    for i in range(len(references) - 1, -1,-1):  # 同样不能用正序循环，for i in range(0,len(alist)), 用了remove()之后，len(alist)是动态的，会产生列表下标越界错误
        if references[i] == '':
            references.remove('')  # 从左往右删除首次出现的值为‘d'的元素

# 计算余弦相似度
def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score


def sent_tokenize(text):
    sents = nlkt_sent_tokenize(text)
    sents_filtered = []
    for s in sents:
        sents_filtered.append(s)
    return sents_filtered


def cleanup_sentences(text):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    sentences_cleaned = []
    for sent in sentences:
        words = nlkt_word_tokenize(sent)
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if not w.lower() in stop_words]
        words = [w.lower() for w in words]
        sentences_cleaned.append(" ".join(words))
    return sentences_cleaned


def get_tf_idf(sentences):
    vectorizer = CountVectorizer()
    sent_word_matrix = vectorizer.fit_transform(sentences)

    transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
    tfidf = transformer.fit_transform(sent_word_matrix)
    tfidf = tfidf.toarray()

    centroid_vector = tfidf.sum(0)
    centroid_vector = np.divide(centroid_vector, centroid_vector.max())

    feature_names = vectorizer.get_feature_names_out()

    relevant_vector_indices = np.where(centroid_vector > 0.3)[0]

    word_list = list(np.array(feature_names)[relevant_vector_indices])
    return word_list


# 填充字向量
# 这个词向量是一个查找表，用于获取质心和句子嵌入表示。
def word_vectors_cache(sentences, embedding_model):
    word_vectors = dict()
    for sent in sentences:
        words = nlkt_word_tokenize(sent)
        for w in words:
            word_vectors.update({w: embedding_model.wv[w]})
    return word_vectors


# 基于词向量和的句子嵌入表示
def build_embedding_representation(words, word_vectors, embedding_model):
    embedding_representation = np.zeros(embedding_model.vector_size, dtype="float32")
    word_vectors_keys = set(word_vectors.keys())
    count = 0
    for w in words:
        if w in word_vectors_keys:
            embedding_representation = embedding_representation + word_vectors[w]
            count += 1
    if count != 0:
        embedding_representation = np.divide(embedding_representation, count)
    return embedding_representation


def summarize(text, emdedding_model):
    raw_sentences = sent_tokenize(text)
    clean_sentences = cleanup_sentences(text)
    '''
    for i, s in enumerate(raw_sentences):
        print(i, s)
    for i, s in enumerate(clean_sentences):
        print(i, s)
    '''
    centroid_words = get_tf_idf(clean_sentences)

    ##print(len(centroid_words), centroid_words)

    word_vectors = word_vectors_cache(clean_sentences, emdedding_model)

    # 质心嵌入表示
    centroid_vector = build_embedding_representation(centroid_words, word_vectors, emdedding_model)
    sentences_scores = []
    for i in range(len(clean_sentences)):
        scores = []
        words = clean_sentences[i].split()

        # 句子嵌入表示
        sentence_vector = build_embedding_representation(words, word_vectors, emdedding_model)

        # 句子嵌入和质心嵌入的余弦相似度
        score = similarity(sentence_vector, centroid_vector)
        sentences_scores.append((i, raw_sentences[i], score, sentence_vector))
    sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
    ##for s in sentence_scores_sort:
        ##print(s[0], s[1], s[2])
    count = 0
    sentences_summary = []
    # 处理冗余
    for s in sentence_scores_sort:
        if count > 100:
            break
        include_flag = True
        for ps in sentences_summary:
            sim = similarity(s[3], ps[3])
            if sim > 0.95:
                include_flag = False
        if include_flag:
            sentences_summary.append(s)
            count += len(s[1].split())

        sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)

    summary = "\n".join([s[1] for s in sentences_summary])
    for s in sentence_scores_sort:
       print(s[0], s[1], s[2])
    return sentence_scores_sort


from Document import  Document

def extractSummary(documents:Document,**kwargs):
    """
       extractSummary chooses a subset of the input documents to serve as a summary
       :param document: Input documents, specified as a tokenizedDocument array.
       :**kwargs:Name=Value pairs
       :return: (summary,scores),a tuple include two lists
                Summary document scores, returned as a list, where scores(i) is the score
                of the jth summary document according to the 'ScoringMethod' option
       """
    eliminata_list_token(documents.tokens())
    mytest=" ".join(documents.tokens())
    clean_sentences = cleanup_sentences(mytest)
    words = []
    for sent in clean_sentences:
        words.append(nlkt_word_tokenize(sent))
    model = Word2Vec(words, min_count=1, sg=1)
    sum_score=summarize(mytest, model)
    summary=[]
    score=[]
    for s in sum_score:
       summary.append(s[1])
       score.append(s[2])
       print(s[0], s[1], s[2])
       break
    summary = " ".join(summary)
    return (summary,score)


def rakeKeywords(documents:Document,**kwargs):
    """
        extracts keywords and respective scores using the Rapid Automatic Keyword Extraction (RAKE) algorithm.
       :param document: Input documents, specified as a tokenizedDocument array.
       :**kwargs:Name=Value pairs
       :return: tb1 :Extracted keywords and scores, returned as a table with the following variables
    """
    from rake_nltk import Rake
    r = Rake()
    mytest=" ".join(documents.tokens())
    r.extract_keywords_from_text(mytest)
    p_s=r.get_ranked_phrases_with_scores()
    print(type(p_s))
    key_w=[]
    score=[]
    tb1={}
    for p_s_tule in p_s:
        if p_s_tule[1] not in key_w:
            key_w.append(p_s_tule[1])
            score.append(p_s_tule[0])
            tb1[p_s_tule[1]]=p_s_tule[0]
    return tb1


def textrankKeyword(documents:Document,**kwargs):
    """
      extracts keywords and respective scores using TextRank.
      :param document: Input documents, specified as a tokenizedDocument array.
      :**kwargs:Name=Value pairs
      :return: tb1 .Extracted keywords and scores, returned as a dictionary with the following variables
    """
    import spacy
    import pytextrank
    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")
    eliminata_list_token(documents.tokens())
    mytest = " ".join(documents.tokens())
    print(mytest)
    # add PyTextRank to the spaCy pipeline
    nlp.add_pipe("textrank")
    doc = nlp(mytest)

    tb1 = {}
    # examine the top-ranked phrases in the document
    for phrase in doc._.phrases:
        tb1[phrase.text] = phrase.rank
    return tb1

def  bleuEvaluationScore(candidate:list,references:list,**kwargs):
    """
         The BiLingual Evaluation Understudy (BLEU) scoring algorithm evaluates the similarity between a candidate document and a
         collection of reference documents. Use the BLEU score to evaluate the quality of document translation and summarization
         models.
         :param candidate: Candidate list, specified as a tokenizedDocument scalar,
         :param references:Reference list, specified as a tokenizedDocument array,
         :return: score, the BLEU similarity score between the specified candidate document and the reference documents.
    """
    from nltk.translate.bleu_score import sentence_bleu
    eliminata_list_token(references)
    eliminata_list_token(candidate)
    references=[references]
    score = sentence_bleu(references, candidate)
    return score

def rougeEvaluationScore(candidate:str,references:str,**kwargs):
    """
         The Recall-Oriented Understudy for Gisting Evaluation (ROUGE) scoring algorithm evaluates the similarity between
         a candidate document and a collection of reference documents. Use the ROUGE score to evaluate the quality of
         document translation and summarization models.
         :param candidate: Candidate list, specified as a tokenizedDocument scalar,
         :param references:Reference list, specified as a tokenizedDocument array,
         :return: score, the ROUGH similarity score between the specified candidate document and the reference documents.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(references,candidate)

    return scores

def bm25Similarity(document,queries=None):
    """
          Use bm25Similarity to calculate document similarities.
         :param document:Input documents, specified as a tokenizedDocument array,
                          a string array of words, or a cell array of character vectors.
         :param queries:Reference list, specified as a tokenizedDocument array,
         :return: score, if queries is None The score in similarities(i,j) represents the similarity between documents(i) and documents(j).
                         if queries isn't None The score in similarities(i,j) represents the similarity between documents(i) and queries(j).
    """
    from gensim import corpora, similarities
    from gensim.summarization.bm25 import BM25
    scores=[]
    if queries is None:
        doc_list = []
        for doc in document:
            doc_list.append(doc.split())
        dictionary = corpora.Dictionary(doc_list)
        corpus = [dictionary.doc2bow(doc) for doc in doc_list]
        bm25 = BM25(corpus)

        for s in document:
            query_bow = dictionary.doc2bow(s.lower().split())
            line_scores = bm25.get_scores(query_bow)
            scores.append(line_scores)
        return scores
    else:
        doc_list = []
        for doc in document:
            doc_list.append(doc.split())
        dictionary = corpora.Dictionary(doc_list)
        corpus = [dictionary.doc2bow(doc) for doc in doc_list]
        bm25 = BM25(corpus)

        for s in queries:
            query_bow = dictionary.doc2bow(s.lower().split())
            line_scores = bm25.get_scores(query_bow)
            scores.append(line_scores)
        return scores

def cosineSimilarity(documents,queries=None):
    """
        cosineSimilarity(documents) returns the pairwise cosine similarities for the specified
        documents using the tf-idf matrix derived from their word counts.
        :param document:Input documents, specified as a list of str,
                        a string array of words, or a cell array of character vectors.
        :param queries:Reference list, specified as a list of str,
        :return: score, if queries is None The score in similarities(i,j) represents the similarity between documents(i) and documents(j).
                     if queries isn't None The score in similarities(i,j) represents the similarity between documents(i) and queries(j).
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if queries is None:
        countVectorizer = CountVectorizer()  # 若要过滤停用词，可在初始化模型时设置
        doc_term_matrix = countVectorizer.fit_transform(documents)  # 得到的doc_term_matrix是一个csr的稀疏矩阵
        # doc_term_matrix[doc_term_matrix>0]=1 #将出现次数大于0的token置1
        # doc_term_matrix.todense()#将稀疏矩阵转化为稠密矩阵
        vocabulary = countVectorizer.vocabulary_  # 得到词汇表

        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(doc_term_matrix)  # 得到的tfidf同样是一个csr的稀疏矩阵
        tfidf_matrix = doc_term_matrix.todense()

        matrix = np.array(tfidf_matrix)
        matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))

        line = matrix.shape[0]
        cos_sim = np.zeros((line, line))
        for i in range(line):
            for j in range(line):
                cos_sim[i][j] = cosine_similarity(matrix[i].reshape(1, -1), matrix[j].reshape(1, -1))


        return cos_sim

    else:
        line=len(documents)
        q_line=len(queries)
        print(line,q_line)
        documents=documents+queries
        print(documents)
        countVectorizer = CountVectorizer()  # 若要过滤停用词，可在初始化模型时设置
        doc_term_matrix = countVectorizer.fit_transform(documents)  # 得到的doc_term_matrix是一个csr的稀疏矩阵
        # doc_term_matrix[doc_term_matrix>0]=1 #将出现次数大于0的token置1
        # doc_term_matrix.todense()#将稀疏矩阵转化为稠密矩阵
        vocabulary = countVectorizer.vocabulary_  # 得到词汇表

        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(doc_term_matrix)  # 得到的tfidf同样是一个csr的稀疏矩阵
        tfidf_matrix = doc_term_matrix.todense()

        matrix = np.array(tfidf_matrix)
        matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))

        cos_sim = np.zeros((line, q_line))
        for i in range(line):
            for j in range(line,line+q_line):
                cos_sim[i][j-line] = cosine_similarity(matrix[i].reshape(1, -1), matrix[j].reshape(1, -1))


        return cos_sim