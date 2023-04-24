from basis_embedding import *
import basis_embedding
_KEY_TYPES = (str, int, np.integer)
_EXTENDED_KEY_TYPES = (str, int, np.integer, np.ndarray)
def _ensure_list(value):
    #确保输入为列表
    if value is None:
        return []
    if isinstance(value, _KEY_TYPES) or (isinstance(value, ndarray) and len(value.shape) == 1):
        return [value]
    return value

def fastTextWordEmbedding():
    """
    Read the pre-trained model and return it
    :return: An object of the Vector class, containing the trained model
    """
    return readWordEmbedding("my_model1.bin")


def doc2sequence(source,documents,PaddingDirection=None,PaddingValue=0,Length='longest'):
    """
        Convert documents to sequences for deep learning
        :param source: A document
        :param documents: documents, e.g. ['thou art bud never art art','art art bud bud'],'thou art bud never art art','test.txt'
        :param PaddingDirection: the direction for padding, e.g. 'left' ,'right' , None
        :param Paddingvalue: the value for padding, e.g. 0
        :param Length: the length of list, e.g. 'longest','shortest',100
        :return: the sequences for input
        """
    list = _ensure_list(documents)
    ans_list=[]
    ans = []
    if isinstance(source, dictionary):
        for temp in list:
            ans_list=[source.doc2bow(temp.lower().split())]
            for li in ans_list:
                t = []
                for temp, temp1 in li:
                    t += [temp]
                ans.append(t)
    elif isinstance(source, Vectors):
        for temp in list:
            ans_list = [source.doc2bow(temp.lower().split())]
            for li in ans_list:
                t = []
                for temp, temp1 in li:
                    t += [temp]
                ans.append(t)
    else:
        print("unknown input")
    maxnum=0
    if Length=='longest':
        for temp in ans:
            if maxnum < len(temp):
                maxnum = len(temp)
    elif Length=='shortest':
        maxnum=len(ans[0])
        for temp in ans:
            if maxnum > len(temp):
                maxnum = len(temp)
    else:
        maxnum=int(Length)
    if PaddingDirection=='left':
        for i in range(0,len(ans)):
            if len(ans[i])<maxnum:
                ans[i]=[PaddingValue]*(maxnum-len(ans[i]))+ans[i]
    elif PaddingDirection=='right':
        for i in range(0, len(ans)):
            if len(ans[i]) < maxnum:
                ans[i] =  ans[i]+[PaddingValue] * (maxnum - len(ans[i]))
    return ans


def readWordEmbedding(filename):
    """
        Read word embedding from file
        :param filename: a file name, e.g. 'my_model1.bin'
        :return: An object of the Vector class, containing the trained model
        """
    #总起,负责处理bin文件的读入,格式为 词数 维度数 后面再接词和向量
    counts = None
    with open(filename, 'rb') as fin:
        header = any2unicode(fin.readline(), encoding='utf8')
        vocab_size, vector_size = [int(x) for x in header.split()]  # 读入词向量的行数以及维度
        kv = Vectors(vector_size, vocab_size, dtype=REAL)
        for line_no in range(vocab_size):
            line = fin.readline()
            if line == b'':
                print("file maybe damaged")

            parts = to_unicode(line.rstrip(), encoding='utf8', errors='').split(" ")
            word, weights = parts[0], [str(x) for x in parts[1:]]
            basis_embedding._add_word_to_kv(kv, counts, word, weights, vocab_size)
    return kv
def writeWordEmbedding(emb,filename):
    """
        Write word embedding file
        :param emb: An object of the Vector class, containing the trained model, e.g. my_model
        :param filename: Name of the file to be written, e.g. 'my_model1.bin'
        """
    #向文件写入词向量,格式为 词数 维度数 后面再接词和向量
    mode = 'wb'
    if 'count' in emb.expandos:
        store_order_vocab_keys = sorted(emb.key_to_index.keys(), key=lambda k: -emb.get_vecattr(k,'count'))
    else:
        store_order_vocab_keys = emb.index_to_key
    assert (len(emb.index_to_key), emb.vector_size) ==emb.vectors.shape
    index_id_count = 0
    for i, val in enumerate(emb.index_to_key):
        if i != val:
            break
        index_id_count += 1
    keys_to_write = itertools.chain(range(0, index_id_count), store_order_vocab_keys)
    with open(filename, mode) as fout:
        fout.write(f"{len(emb.vectors)} {emb.vector_size}\n".encode('utf8'))
        for key in keys_to_write:
            key_vector = word2vec(emb,key)
            fout.write(f"{key} {' '.join(repr(val) for val in key_vector)}\n".encode('utf8'))

def ind2word(emb,lst):
    """
        Map encoding index to word

        :param emb: An object of the Vector class, containing the trained model, e.g. my_model
        :param lst: a list of integers., e.g. 1 , 2 , [1,2,5]
        :return: words, returned as a string vector. ., e.g. ['the'] , ['the','king']
    """
    #根据序号获取词汇
    ans_list=[]
    lst=_ensure_list(lst)
    finlst=list(enumerate(emb.index_to_key,1))
    for num in lst:
        ans_list+=[finlst[num][1]]
    return ans_list

def isVocabularyWord(emb,lst):
    """
        Test if word is member of word embedding or encoding

        :param emb: An object of the Vector class, containing the trained model, e.g. my_model
        :param lst: a list of words., e.g. ['the'] , ['the','king']
        :return: Whether the word at the corresponding position exists in the vector e.g. [1] , [0,1]
        """
    #判断是否为单词表中的单词
    ans_list = []
    lst = _ensure_list(lst)
    for key in lst:
        val = emb.key_to_index.get(key, -1)
        if val >= 0:
            ans_list+=[1]
        else:
            ans_list+=[0]
    return ans_list

def word2ind(emb, lst):
    """
        Map word to encoding index

        :param emb: An object of the Vector class, containing the trained model, e.g. my_model
        :param lst: a list of words., e.g. ['the'] , ['the','king']

        :return: List of serial numbers of corresponding words e.g. [10] , [230,14]
        """
    #找到词汇对应的序号
    ans_list = []
    lst=_ensure_list(lst)
    for key in lst:
        val = emb.key_to_index.get(key, -1)
        if val >= 0:
            ans_list+=[val]
        else:
            print("can not find word",key)
    return ans_list
def wordEncoding(source,**kwargs):
    ans=[]
    if isinstance(source, str):
        # 输入来源为txt文件
        sen = sen_get(source)
        a=dictionary(documents=sen)
    elif isinstance(source, list):
        # 输入为列表或长句子
        texts = [
            [word for word in document.lower().split()]
            for document in source
        ]
        a = dictionary(documents=texts)
    for value in a.token2id:
        ans+=[value]
    return a
def word2vec(emb,str):
    """
        Map word to embedding vector

        :param emb: An object of the Vector class, containing the trained model, e.g. my_model
        :param str: a list of words., e.g. 'the'
        :return: The corresponding vector of the word
        """
    #找到词汇对应的向量
    a=emb.get_vector(str)
    return a

def vec2word(emb,vec,k=1):
    """
        Map embedding vector to word

        :param emb: An object of the Vector class, containing the trained model, e.g. my_model
        :param vec: The corresponding vector of the word, e.g. [0,0.13,0.5]
        :param k: The num to return, e.g. 1,5
        :return: The word that is closest to this vector , return a list if k is not 1, e.g. 'king' , ['king','queen','princess']
        """
    #将向量转变为最接近的词汇
    clip_start = 0

    if isinstance(k, Integral) and k < 1:
        return []
    positive = _ensure_list(vec)
    emb.fill_norms()
    clip_end = len(emb.vectors)

    positive = [
        (item, 1.0) if isinstance(item, _EXTENDED_KEY_TYPES) else item
        for item in positive
    ]
    all_keys, mean = set(), []
    for key, weight in positive:
        if isinstance(key, ndarray):
            mean.append(weight * key)
        else:
            mean.append(weight * emb.get_vector(key, norm=True))
            if emb.has_index_for(key):
                all_keys.add(emb.get_index(key))
    if not mean:
        print("no input")
    mean = unitvec(array(mean).mean(axis=0)).astype(REAL)

    dists = dot(emb.vectors[clip_start:clip_end], mean) / emb.norms[clip_start:clip_end]
    if not k:
        return dists
    best = argsort(dists, topn=k + len(all_keys), reverse=True)

    result = [
        (emb.index_to_key[sim + clip_start])
        for sim in best
    ]

    if k:
        result = result[:k]
    if k==1:
        return result[0]
    return result
def wordEmbeddingLayer(dimension,numWords,**kwargs):
    """
        Word embedding layer for deep learning networks

        :param Dimension: Dimension , e.g. 100
        :param NumWords: Number of words in model, e.g. 200
        :param WeightsInitializer: Function to initialize weights, e.g. 'he'
        :param Weights: Layer weights
        :param WeightLearnRateFactor: Learning rate factor for weights, e.g. 1
        :param WeightL2Factor: L2 regularization factor for weights, e.g. 1
        :return: layer
        """
    a=my_layer()
    a.start(Dimension=dimension,NumWords=numWords,**kwargs)
    return a

def trainWordEmbedding(source,**kwargs):
    """
        Train word embedding

        :param source: A document or some sentences, e.g. 'test.txt' , ["a b c d e","in of and or"]
        :param Dimension:Dimension of word embedding, e.g. 100
        :param Window: Size of context window, e.g. 5
        :param Model: model to use, e.g. 'cbow'
        :param DiscardFactor: Factor to determine word discard rate, e.g. 1e-4
        :param LossFunction:Loss function, e.g. 'ns'
        :param NumNegativeSamples: Number of negative samples, e.g. 5
        :param NumEpochs: Number of epochs, e.g. 5
        :param MinCount: Minimum count of words, e.g. 5
        :param NGramRange: Inclusive range for subword n-grams, e.g. [3,6]
        :param InitialLearnRate: Initial learn rate, e.g. 0.05
        :param UpdateRate:  Rate for updating learn rate, e.g. 100
        :return: Output word embedding, returned as a Vector object.
        """
    if isinstance(source, str):
        #输入来源为txt文件
        sen=sen_get(source)
        #print(sen)
        a=Word2Vec(sentences=sen,**kwargs)
        return a.wv
    elif isinstance(source, list):
        #输入为列表或长句子
        a=Word2Vec(sentences=source,**kwargs)
        return a.wv

#
# documents = [
#      "fairest creatures desire increase thereby beautys rose might never die riper time decease tender heir might bear memory thou contracted thine own bright eyes feedst thy lights flame selfsubstantial fuel making famine abundance lies thy self thy foe thy sweet self cruel thou art worlds fresh ornament herald gaudy spring thine own bud buriest thy content tender churl makst waste niggarding pity world else glutton eat worlds due grave thee",
#     "a a a a a a a a a a a forty winters shall besiege thy brow dig deep trenches thy beautys field thy youths proud livery gazed tatterd weed small worth held asked thy beauty lies treasure thy lusty days say thine own deep sunken eyes alleating shame thriftless praise praise deservd thy beautys thou couldst answer fair child mine shall sum count make old excuse proving beauty succession thine new made thou art old thy blood warm thou feelst cold",
#     "look thy glass tell face thou viewest time face form another whose fresh repair thou renewest thou dost beguile world unbless mother fair whose uneard womb disdains tillage thy husbandry fond tomb selflove stop posterity thou art thy mothers glass thee calls back lovely april prime thou windows thine age shalt despite wrinkles thy golden time thou live rememberd die single thine image dies thee",
#     "unthrifty loveliness why dost thou spend upon thy self thy beautys legacy natures bequest gives nothing doth lend frank lends free beauteous niggard why dost thou abuse bounteous largess thee give profitless usurer why dost thou great sum sums yet canst live traffic thy self alone thou thy self thy sweet self dost deceive nature calls thee gone acceptable audit canst thou leave thy unused beauty tombed thee lives th executor",
#     "hours gentle work frame lovely gaze every eye doth dwell play tyrants same unfair fairly doth excel neverresting time leads summer hideous winter confounds sap checked frost lusty leaves quite gone beauty oersnowed bareness every summers distillation left liquid prisoner pent walls glass beautys effect beauty bereft nor nor remembrance flowers distilld though winter meet leese show substance still lives sweet",
#     "let winters ragged hand deface thee thy summer ere thou distilld make sweet vial treasure thou place beautys treasure ere selfkilld forbidden usury happies pay willing loan thats thy self breed another thee ten times happier ten ten times thy self happier thou art ten thine ten times refigurd thee death thou shouldst depart leaving thee living posterity selfwilld thou art fair deaths conquest make worms thine heir",
#     "lo orient gracious light lifts up burning head eye doth homage newappearing sight serving looks sacred majesty climbd steepup heavenly hill resembling strong youth middle age yet mortal looks adore beauty still attending golden pilgrimage highmost pitch weary car like feeble age reeleth day eyes fore duteous converted low tract look another way thou thyself outgoing thy noon unlookd diest unless thou get son",
#     "music hear why hearst thou music sadly sweets sweets war joy delights joy why lovst thou thou receivst gladly else receivst pleasure thine annoy true concord welltuned sounds unions married offend thine ear sweetly chide thee confounds singleness parts thou shouldst bear mark string sweet husband another strikes mutual ordering resembling sire child happy mother pleasing note sing whose speechless song many seeming sings thee thou single wilt prove none",
#     "fear wet widows eye thou consumst thy self single life ah thou issueless shalt hap die world wail thee like makeless wife world thy widow still weep thou form thee hast left behind every private widow well keep childrens eyes husbands shape mind look unthrift world doth spend shifts place still world enjoys beautys waste hath world end kept unused user destroys love toward others bosom sits murdrous shame commits",
#     "shame deny thou bearst love thy self art unprovident grant thou wilt thou art belovd many thou none lovst evident thou art possessd murderous hate gainst thy self thou stickst conspire seeking beauteous roof ruinate repair thy chief desire o change thy thought change mind shall hate fairer lodgd gentle love thy presence gracious kind thyself least kindhearted prove make thee another self love beauty still live thine thee"
# ]
# dictionary2=wordEncoding(documents)
# dictionary1=wordEncoding('test_wordEncoding.txt')
# print(dictionary1)
# print(dictionary1)