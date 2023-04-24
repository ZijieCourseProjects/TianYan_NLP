import heapq
import itertools
import math
import os
import platform
import re
import sys
import threading
import unicodedata
from collections import defaultdict, namedtuple
from copy import deepcopy
from datetime import datetime
from numbers import Integral
from queue import Queue
from timeit import default_timer

import numpy as np
import scipy.sparse
from numpy import (
    dot, float32 as REAL, array, zeros, vstack,
    ndarray, dtype, frombuffer,
)
from scipy.linalg import get_blas_funcs
from smart_open import open

# import model_load
_KEY_TYPES = (str, int, np.integer)
_EXTENDED_KEY_TYPES = (str, int, np.integer, np.ndarray)
PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)


class Vectors():
    # 保存词向量
    def __init__(self, vector_size, count=0, dtype=np.float32):
        self.vector_size = vector_size
        self.index_to_key = [None] * count
        self.next_index = 0
        self.key_to_index = {}
        self.vectors = zeros((count, vector_size), dtype=dtype)
        self.norms = None
        self.expandos = {}

    def __len__(self):
        return len(self.index_to_key)

    def doc2bow(self, document):
        counter = defaultdict(int)
        for w in document:
            counter[w if isinstance(w, str) else str(w, 'utf-8')] += 1
        token2id = self.key_to_index
        result = {token2id[w]: freq for w, freq in counter.items() if w in token2id}
        result = sorted(result.items())
        return result

    def fill_norms(self, force=False):
        if self.norms is None or force:
            self.norms = np.linalg.norm(self.vectors, axis=1)

    def resize_vectors(self, seed=0):
        target_shape = (len(self.index_to_key), self.vector_size)
        self.vectors = prep_vectors(target_shape, prior_vectors=self.vectors, seed=seed)
        self.allocate_vecattrs()
        self.norms = None

    def sort_by_descending_frequency(self):
        if not len(self):
            return  # noop if empty
        count_sorted_indexes = np.argsort(self.expandos['count'])[::-1]
        self.index_to_key = [self.index_to_key[idx] for idx in count_sorted_indexes]
        self.allocate_vecattrs()
        for k in self.expandos:
            # Use numpy's "fancy indexing" to permutate the entire array in one step.
            self.expandos[k] = self.expandos[k][count_sorted_indexes]
        if len(self.vectors):
            self.vectors = self.vectors[count_sorted_indexes]
        self.key_to_index = {word: i for i, word in enumerate(self.index_to_key)}

    def allocate_vecattrs(self, attrs=None, types=None):
        # 分配,准备词向量读入
        if attrs is None:
            attrs = list(self.expandos.keys())
            types = [self.expandos[attr].dtype for attr in attrs]
        target_size = len(self.index_to_key)
        for attr, t in zip(attrs, types):
            if t is int:
                t = np.int64
            if t is str:
                t = object
            if attr not in self.expandos:
                self.expandos[attr] = np.zeros(target_size, dtype=t)
                continue
            prev_expando = self.expandos[attr]
            if len(prev_expando) == target_size:
                continue  # no resizing necessary
            prev_count = len(prev_expando)
            self.expandos[attr] = np.zeros(target_size, dtype=prev_expando.dtype)
            self.expandos[attr][: min(prev_count, target_size), ] = prev_expando[: min(prev_count, target_size), ]

    def has_index_for(self, key):
        # 存在该词汇
        return self.get_index(key, -1) >= 0

    def add_vectors(self, keys, weights, extras=None, replace=False):
        # 向类中添加词和对应向量
        if isinstance(keys, _KEY_TYPES):
            keys = [keys]
            weights = np.array(weights).reshape(1, -1)
        elif isinstance(weights, list):
            weights = np.array(weights)
        if extras is None:
            extras = {}
        self.allocate_vecattrs(extras.keys(), [extras[k].dtype for k in extras.keys()])
        in_vocab_mask = np.zeros(len(keys), dtype=bool)
        for idx, key in enumerate(keys):
            if key in self:
                in_vocab_mask[idx] = True
        for idx in np.nonzero(~in_vocab_mask)[0]:
            key = keys[idx]
            self.key_to_index[key] = len(self.index_to_key)
            self.index_to_key.append(key)
        self.vectors = vstack((self.vectors, weights[~in_vocab_mask].astype(self.vectors.dtype)))
        for attr, extra in extras:
            self.expandos[attr] = np.vstack((self.expandos[attr], extra[~in_vocab_mask]))
        if replace:
            in_vocab_idxs = [self.get_index(keys[idx]) for idx in np.nonzero(in_vocab_mask)[0]]
            self.vectors[in_vocab_idxs] = weights[in_vocab_mask]
            for attr, extra in extras:
                self.expandos[attr][in_vocab_idxs] = extra[in_vocab_mask]

    def set_vecattr(self, key, attr, val):
        self.allocate_vecattrs(attrs=[attr], types=[type(val)])
        index = self.get_index(key)
        self.expandos[attr][index] = val

    def get_vecattr(self, key, attr):
        index = self.get_index(key)
        return self.expandos[attr][index]

    def add_vector(self, key, vector):
        # 添加向量
        target_index = self.next_index
        if target_index >= len(self.vectors) or self.index_to_key[target_index] is not None:
            target_index = len(self.vectors)
            self.add_vectors([key], [vector])
            self.allocate_vecattrs()
            self.next_index = target_index + 1
        else:
            self.index_to_key[target_index] = key
            self.key_to_index[key] = target_index
            self.vectors[target_index] = vector
            self.next_index += 1
        return target_index

    def get_index(self, key, default=None):
        # 获取对应index
        val = self.key_to_index.get(key, -1)
        if val >= 0:
            return val
        elif isinstance(key, (int, np.integer)) and 0 <= key < len(self.index_to_key):
            return key
        elif default is not None:
            return default
        else:
            print("can not find word", key)

    def get_vector(self, key):
        # 获取对应向量
        index = self.get_index(key)
        result = self.vectors[index]

        result.setflags(write=False)
        return result


def prune_vocab(vocab, min_reduce, trim_rule=None):
    result = 0
    old_len = len(vocab)
    for w in list(vocab):  # make a copy of dict's keys
        if not keep_vocab_item(w, vocab[w], min_reduce, trim_rule):  # vocab[w] <= min_reduce:
            result += vocab[w]
            del vocab[w]
    return result


def prep_vectors(target_shape, prior_vectors=None, seed=0, dtype=REAL):
    if prior_vectors is None:
        prior_vectors = np.zeros((0, 0))
    if prior_vectors.shape == target_shape:
        return prior_vectors
    target_count, vector_size = target_shape
    rng = np.random.default_rng(seed=seed)
    new_vectors = rng.random(target_shape, dtype=dtype)  # [0.0, 1.0)
    new_vectors *= 2.0  # [0.0, 2.0)
    new_vectors -= 1.0  # [-1.0, 1.0)
    new_vectors /= vector_size
    new_vectors[0:prior_vectors.shape[0], 0:prior_vectors.shape[1]] = prior_vectors
    return new_vectors


def keep_vocab_item(word, count, min_count, trim_rule=None):
    default_res = count >= min_count
    RULE_DEFAULT = 0
    RULE_DISCARD = 1
    RULE_KEEP = 2
    if trim_rule is None:
        return default_res
    else:
        rule_res = trim_rule(word, count, min_count)
        if rule_res == RULE_KEEP:
            return True
        elif rule_res == RULE_DISCARD:
            return False
        else:
            return default_res


class dictionary():
    def __init__(self, documents=None, prune_at=2000000):
        self.token2id = {}
        self.id2token = {}
        self.cfs = {}
        self.dfs = {}
        self.num_docs = 0
        self.num_pos = 0
        self.num_nnz = 0
        if documents is not None:
            self.add_documents(documents, prune_at=prune_at)

    def __len__(self):
        return len(self.token2id)

    def add_documents(self, documents, prune_at=2000000):
        for docno, document in enumerate(documents):
            if docno % 10000 == 0:
                if prune_at is not None and len(self) > prune_at:
                    self.filter_extremes(no_below=5, no_above=1.0, keep_n=prune_at)
            self.doc2bow(document, allow_update=True)

    def filter_extremes(self, no_below=5, no_above=0.5, keep_n=100000, keep_tokens=None):
        no_above_abs = int(no_above * self.num_docs)

        if keep_tokens:
            keep_ids = {self.token2id[v] for v in keep_tokens if v in self.token2id}
            good_ids = [
                v for v in self.token2id.values()
                if no_below <= self.dfs.get(v, 0) <= no_above_abs or v in keep_ids
            ]
            good_ids.sort(key=lambda x: self.num_docs if x in keep_ids else self.dfs.get(x, 0), reverse=True)
        else:
            good_ids = [
                v for v in self.token2id.values()
                if no_below <= self.dfs.get(v, 0) <= no_above_abs
            ]
            good_ids.sort(key=self.dfs.get, reverse=True)
        if keep_n is not None:
            good_ids = good_ids[:keep_n]
        self.filter_tokens(good_ids=good_ids)

    def doc2bow(self, document, allow_update=False):
        counter = defaultdict(int)
        for w in document:
            counter[w if isinstance(w, str) else str(w, 'utf-8')] += 1

        token2id = self.token2id
        if allow_update:
            missing = sorted(x for x in counter.items() if x[0] not in token2id)
            if allow_update:
                for w, _ in missing:
                    token2id[w] = len(token2id)
        result = {token2id[w]: freq for w, freq in counter.items() if w in token2id}

        if allow_update:
            self.num_docs += 1
            self.num_pos += sum(counter.values())
            self.num_nnz += len(result)
            for tokenid, freq in result.items():
                self.cfs[tokenid] = self.cfs.get(tokenid, 0) + freq
                self.dfs[tokenid] = self.dfs.get(tokenid, 0) + 1

        result = sorted(result.items())
        return result

    def filter_tokens(self, bad_ids=None, good_ids=None):
        if bad_ids is not None:
            bad_ids = set(bad_ids)
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid not in bad_ids}
            self.cfs = {tokenid: freq for tokenid, freq in self.cfs.items() if tokenid not in bad_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid not in bad_ids}
        if good_ids is not None:
            good_ids = set(good_ids)
            self.token2id = {token: tokenid for token, tokenid in self.token2id.items() if tokenid in good_ids}
            self.cfs = {tokenid: freq for tokenid, freq in self.cfs.items() if tokenid in good_ids}
            self.dfs = {tokenid: freq for tokenid, freq in self.dfs.items() if tokenid in good_ids}
        self.compactify()

    def compactify(self):
        # build mapping from old id -> new id
        idmap = dict(zip(sorted(self.token2id.values()), range(len(self.token2id))))

        # reassign mappings to new ids
        self.token2id = {token: idmap[tokenid] for token, tokenid in self.token2id.items()}
        self.id2token = {}
        self.dfs = {idmap[tokenid]: freq for tokenid, freq in self.dfs.items()}
        self.cfs = {idmap[tokenid]: freq for tokenid, freq in self.cfs.items()}


def any2unicode(text, encoding='utf8', errors='strict'):
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)


def _add_word_to_kv(kv, counts, word, weights, vocab_size):
    # 向类中补充词汇
    if kv.has_index_for(word):
        return
    word_id = kv.add_vector(word, weights)
    if counts is None:
        word_count = vocab_size - word_id
    elif word in counts:
        word_count = counts[word]
    else:
        word_count = None
    kv.set_vecattr(word, 'count', word_count)


def simple_preprocess(doc, deacc=False, min_len=2, max_len=15):
    tokens = [
        token for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def tokenize(text, lowercase=False, deacc=False, encoding='utf8', errors="strict", to_lower=False, lower=False):
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    return simple_tokenize(text)


def deaccent(text):
    if not isinstance(text, str):
        # assume utf8 for byte strings, use default (strict) error handling
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = ''.join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def tokenize(text, lowercase=False, deacc=False, encoding='utf8', errors="strict", to_lower=False, lower=False):
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    return simple_tokenize(text)


def simple_tokenize(text):
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def _add_bytes_to_kv(kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors):
    start = 0
    processed_words = 0
    bytes_per_vector = vector_size * dtype(REAL).itemsize
    max_words = vocab_size - kv.next_index
    assert max_words > 0
    for _ in range(max_words):
        i_space = chunk.find(b' ', start)
        i_vector = i_space + 1

        if i_space == -1 or (len(chunk) - i_vector) < bytes_per_vector:
            break

        word = chunk[start:i_space].decode("utf-8", errors=unicode_errors)
        word = word.lstrip('\n')
        vector = frombuffer(chunk, offset=i_vector, count=vector_size, dtype=REAL).astype(datatype)
        _add_word_to_kv(kv, counts, word, vector, vocab_size)
        start = i_vector + bytes_per_vector
        processed_words += 1

    return processed_words, chunk[start:]


def _word2vec_read_binary(fin, kv, counts, vocab_size, vector_size, datatype, unicode_errors, binary_chunk_size):
    chunk = b''
    tot_processed_words = 0

    while tot_processed_words < vocab_size:
        new_chunk = fin.read(binary_chunk_size)
        chunk += new_chunk
        processed_words, chunk = _add_bytes_to_kv(
            kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors)
        tot_processed_words += processed_words
        if len(new_chunk) < binary_chunk_size:
            break
    if tot_processed_words != vocab_size:
        print("input error")


def blas(name, ndarray):
    return get_blas_funcs((name,), (ndarray,))[0]


blas_nrm2 = blas('nrm2', np.array([], dtype=float))
blas_scal = blas('scal', np.array([], dtype=float))


def unitvec(vec, norm='l2', return_norm=False):
    supported_norms = ('l1', 'l2', 'unique')
    if norm not in supported_norms:
        print("not a supported norm")
    if scipy.sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        if norm == 'l2':
            veclen = np.sqrt(np.sum(vec.data ** 2))
        if norm == 'unique':
            veclen = vec.nnz
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            vec /= veclen
            if return_norm:
                return vec, veclen
            else:
                return vec
        else:
            if return_norm:
                return vec, 1.0
            else:
                return vec

    if isinstance(vec, np.ndarray):
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            if vec.size == 0:
                veclen = 0.0
            else:
                veclen = blas_nrm2(vec)
        if norm == 'unique':
            veclen = np.count_nonzero(vec)
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            if return_norm:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen
            else:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
        else:
            if return_norm:
                return vec, 1.0
            else:
                return vec

    try:
        first = next(iter(vec))  # is there at least one element?
    except StopIteration:
        if return_norm:
            return vec, 1.0
        else:
            return vec

    if isinstance(first, (tuple, list)) and len(first) == 2:  # gensim sparse format
        if norm == 'l1':
            length = float(sum(abs(val) for _, val in vec))
        if norm == 'l2':
            length = 1.0 * math.sqrt(sum(val ** 2 for _, val in vec))
        if norm == 'unique':
            length = 1.0 * len(vec)
        assert length > 0.0, "sparse documents must not contain any explicit zero entries"
    else:
        print("unknown input type")


def argsort(x, topn=None, reverse=False):
    # 寻找向量最接近的
    x = np.asarray(x)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, 'argpartition'):
        return np.argsort(x)[:topn]
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))


def any2utf8(text, errors='strict', encoding='utf8'):
    if isinstance(text, str):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return str(text, encoding, errors=errors).encode('utf8')


to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):
    if isinstance(text, str):
        return text
    return str(text, encoding, errors=errors)


to_unicode = any2unicode


def _ensure_list(value):
    # 确保输入为列表
    if value is None:
        return []
    if isinstance(value, _KEY_TYPES) or (isinstance(value, ndarray) and len(value.shape) == 1):
        return [value]
    return value


def readWordEmbedding(filename):
    # 总起,负责处理bin文件的读入,格式为 词数 维度数 后面再接词和向量
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
            _add_word_to_kv(kv, counts, word, weights, vocab_size)
    return kv


def writeWordEmbedding(emb, filename):
    # 向文件写入词向量,格式为 词数 维度数 后面再接词和向量
    mode = 'wb'
    if 'count' in emb.expandos:
        store_order_vocab_keys = sorted(emb.key_to_index.keys(), key=lambda k: -emb.get_vecattr(k, 'count'))
    else:
        store_order_vocab_keys = emb.index_to_key
    assert (len(emb.index_to_key), emb.vector_size) == emb.vectors.shape
    index_id_count = 0
    for i, val in enumerate(emb.index_to_key):
        if i != val:
            break
        index_id_count += 1
    keys_to_write = itertools.chain(range(0, index_id_count), store_order_vocab_keys)
    with open(filename, mode) as fout:
        fout.write(f"{len(emb.vectors)} {emb.vector_size}\n".encode('utf8'))
        for key in keys_to_write:
            key_vector = word2vec(emb, key)
            fout.write(f"{key} {' '.join(repr(val) for val in key_vector)}\n".encode('utf8'))


def ind2word(emb, lst):
    # 根据序号获取词汇
    ans_list = []
    lst = _ensure_list(lst)
    finlst = list(enumerate(emb.index_to_key, 1))
    for num in lst:
        ans_list += [finlst[num][1]]
    return ans_list


def isVocabularyWord(emb, lst):
    # 判断是否为单词表中的单词
    ans_list = []
    lst = _ensure_list(lst)
    for key in lst:
        val = emb.key_to_index.get(key, -1)
        if val >= 0:
            ans_list += [1]
        else:
            ans_list += [0]
    return ans_list


def word2ind(emb, lst):
    # 找到词汇对应的序号
    ans_list = []
    lst = _ensure_list(lst)
    for key in lst:
        val = emb.key_to_index.get(key, -1)
        if val >= 0:
            ans_list += [val]
        else:
            print("can not find word", key)
    return ans_list


def word2vec(emb, str):
    # 找到词汇对应的向量
    return emb.get_vector(str)


def vec2word(emb, vec):
    # 将向量转变为最接近的词汇
    topn = 10
    clip_start = 0

    if isinstance(topn, Integral) and topn < 1:
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
    if not topn:
        return dists
    best = argsort(dists, topn=topn + len(all_keys), reverse=True)

    result = [
        (emb.index_to_key[sim + clip_start], float(dists[sim]))
        for sim in best if (sim + clip_start) not in all_keys
    ]
    return result[:topn]


def fastTextWordEmbedding():
    return readWordEmbedding("pretrained_model.bin")


def doc2sequence(source, documents, parm1='PaddingDirection', value1=None, parm2='Paddingvalue', value2=0):
    list = _ensure_list(documents)
    ans = []
    if isinstance(source, dictionary):
        for temp in list:
            ans_list = [source.doc2bow(temp.lower().split())]
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

    if parm1 == 'PaddingDirection' and value1 == 'left':
        maxnum = 0
        for temp in ans:
            if maxnum < len(temp):
                maxnum = len(temp)
        for i in range(0, len(ans)):
            if len(ans[i]) < maxnum:
                ans[i] = [value2] * (maxnum - len(ans[i])) + ans[i]
    elif parm1 == 'PaddingDirection' and value1 == 'right':
        maxnum = 0
        for temp in ans:
            if maxnum < len(temp):
                maxnum = len(temp)
        for i in range(0, len(ans)):
            if len(ans[i]) < maxnum:
                ans[i] = ans[i] + [value2] * (maxnum - len(ans[i]))
    return ans


def call_on_class_only(*args, **kwargs):
    raise AttributeError('This method should be called on a class object.')


class LineSentence:
    def __init__(self, source, max_sentence_length=10000, limit=None):
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        try:

            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length
        except AttributeError:
            # If it didn't work like a file, use it as a string filename
            with open(self.source, 'rb') as fin:
                for line in itertools.islice(fin, self.limit):
                    line = to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield line[i: i + self.max_sentence_length]
                        i += self.max_sentence_length


def train_batch_cbow(model, sentences, alpha, _work, _neu1,
                     compute_loss):  # real signature unknown; restored from __doc__
    pass


def train_batch_sg(model, sentences, alpha, _work, compute_loss):
    pass


def _assign_binary_codes(wv):
    heap = _build_heap(wv)
    if not heap:
        return

    # recurse over the tree, assigning a binary code to each vocabulary word
    max_depth = 0
    stack = [(heap[0], [], [])]
    while stack:
        node, codes, points = stack.pop()
        if node[1] < len(wv):  # node[1] = index
            # leaf node => store its path from the root
            k = node[1]
            wv.set_vecattr(k, 'code', codes)
            wv.set_vecattr(k, 'point', points)
            # node.code, node.point = codes, points
            max_depth = max(len(codes), max_depth)
        else:
            # inner node => continue recursion
            points = np.array(list(points) + [node.index - len(wv)], dtype=np.uint32)
            stack.append((node.left, np.array(list(codes) + [0], dtype=np.uint8), points))
            stack.append((node.right, np.array(list(codes) + [1], dtype=np.uint8), points))


class Word2Vec():
    def __init__(
            self, sentences=None, corpus_file=None, Dimension=100, InitialLearnRate=0.025, Window=5, MinCount=5,
            max_vocab_size=None, sample=1e-3, seed=1, workers=3, DiscardFactor=0.0001, Model='cbow', Verbose=0,
            sg=0, hs=0, NumNegativeSamples=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), UpdateRate=50,
            comment=None, max_final_vocab=None, shrink_windows=True, LossFunction='ns', NGramRange=[3, 6]
    ):
        corpus_iterable = sentences

        self.vector_size = int(Dimension)
        self.workers = int(workers)
        self.epochs = epochs
        self.train_count = 0
        self.total_train_time = 0
        self.batch_words = batch_words

        self.sg = int(sg)
        self.alpha = float(InitialLearnRate)
        self.min_alpha = float(DiscardFactor)

        self.window = int(Window)
        self.shrink_windows = bool(shrink_windows)
        self.random = np.random.RandomState(seed)

        self.hs = int(hs)
        self.negative = int(NumNegativeSamples)
        self.ns_exponent = ns_exponent
        self.cbow_mean = int(cbow_mean)
        self.compute_loss = bool(compute_loss)
        self.running_training_loss = 0
        self.min_alpha_yet_reached = float(InitialLearnRate)
        self.corpus_count = 0
        self.corpus_total_words = 0
        LossFunction = LossFunction + 'hs'
        self.max_final_vocab = max_final_vocab
        self.max_vocab_size = max_vocab_size
        self.min_count = MinCount
        self.sample = sample
        self.sorted_vocab = sorted_vocab
        self.null_word = null_word
        self.cum_table = None  # for negative sampling
        self.raw_vocab = None

        if not hasattr(self, 'wv'):
            self.wv = Vectors(Dimension)
        self.wv.vectors_lockf = np.ones(1, dtype=REAL)
        Model = Model + 'ns'
        self.hashfxn = hashfxn
        self.seed = seed
        if not hasattr(self, 'layer1_size'):
            self.layer1_size = Dimension

        self.comment = comment

        self.load = call_on_class_only

        if corpus_iterable is not None or corpus_file is not None:
            self._check_corpus_sanity(corpus_iterable=corpus_iterable, corpus_file=corpus_file, passes=(epochs + 1))
            self.build_vocab(corpus_iterable=corpus_iterable, corpus_file=corpus_file, trim_rule=trim_rule)
            self.train(
                corpus_iterable=corpus_iterable, corpus_file=corpus_file, total_examples=self.corpus_count,
                total_words=self.corpus_total_words, epochs=self.epochs, start_alpha=self.alpha,
                end_alpha=self.min_alpha, compute_loss=self.compute_loss, callbacks=callbacks)

        self.add_lifecycle_event("created", params=str(self))

    def add_lifecycle_event(self, event_name, **event):
        event_dict = deepcopy(event)
        event_dict['datetime'] = datetime.now().isoformat()
        event_dict['python'] = sys.version
        event_dict['platform'] = platform.platform()
        event_dict['event'] = event_name
        if not hasattr(self, 'lifecycle_events'):
            self.lifecycle_events = []
        if self.lifecycle_events is not None:
            self.lifecycle_events.append(event_dict)

    def _check_training_sanity(self, epochs=0, total_examples=None, total_words=None, **kwargs):
        if (not self.hs) and (not self.negative):
            raise ValueError(
                "You must set either 'hs' or 'negative' to be positive for proper training. "
                "When both 'hs=0' and 'negative=0', there will be no training."
            )
        if not self.wv.key_to_index:  # should be set by `build_vocab`
            raise RuntimeError("you must first build vocabulary before training the model")
        if not len(self.wv.vectors):
            raise RuntimeError("you must initialize vectors before training the model")

        if total_words is None and total_examples is None:
            raise ValueError(
                "You must specify either total_examples or total_words, for proper learning-rate "
                "and progress calculations. "
                "If you've just built the vocabulary using the same corpus, using the count cached "
                "in the model is sufficient: total_examples=model.corpus_count."
            )
        if epochs is None or epochs <= 0:
            raise ValueError("You must specify an explicit epochs count. The usual value is epochs=model.epochs.")

    def _get_next_alpha(self, epoch_progress, cur_epoch):
        start_alpha = self.alpha
        end_alpha = self.min_alpha
        progress = (cur_epoch + epoch_progress) / self.epochs
        next_alpha = start_alpha - (start_alpha - end_alpha) * progress
        next_alpha = max(end_alpha, next_alpha)
        self.min_alpha_yet_reached = next_alpha
        return next_alpha

    def _raw_word_count(self, job):

        return sum(len(sentence) for sentence in job)

    def _job_producer(self, data_iterator, job_queue, cur_epoch=0, total_examples=None, total_words=None):
        job_batch, batch_size = [], 0
        pushed_words, pushed_examples = 0, 0
        next_alpha = self._get_next_alpha(0.0, cur_epoch)
        job_no = 0

        for data_idx, data in enumerate(data_iterator):
            data_length = self._raw_word_count([data])
            if batch_size + data_length <= self.batch_words:
                job_batch.append(data)
                batch_size += data_length
            else:
                job_no += 1
                job_queue.put((job_batch, next_alpha))
                if total_examples:
                    pushed_examples += len(job_batch)
                    epoch_progress = 1.0 * pushed_examples / total_examples
                else:
                    # words-based decay
                    pushed_words += self._raw_word_count(job_batch)
                    epoch_progress = 1.0 * pushed_words / total_words
                next_alpha = self._get_next_alpha(epoch_progress, cur_epoch)
                job_batch, batch_size = [data], data_length
        if job_batch:
            job_no += 1
            job_queue.put((job_batch, next_alpha))
        for _ in range(self.workers):
            job_queue.put(None)

    def zeros_aligned(self, shape, dtype, order='C', align=128):

        nbytes = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
        buffer = np.zeros(nbytes + align, dtype=np.uint8)
        start_index = -buffer.ctypes.data % align
        return buffer[start_index: start_index + nbytes].view(dtype).reshape(shape, order=order)

    def _get_thread_working_mem(self):
        work = self.zeros_aligned(self.layer1_size, dtype=REAL)
        neu1 = self.zeros_aligned(self.layer1_size, dtype=REAL)
        return work, neu1

    def _do_train_job(self, sentences, alpha, inits):
        work, neu1 = inits
        tally = 0

        return tally, self._raw_word_count(sentences)

    def _worker_loop(self, job_queue, progress_queue):

        thread_private_mem = self._get_thread_working_mem()
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker
            data_iterable, alpha = job

            tally, raw_tally = self._do_train_job(data_iterable, alpha, thread_private_mem)

            progress_queue.put((len(data_iterable), tally, raw_tally))  # report back progress
            jobs_processed += 1

    def _log_epoch_progress(
            self, progress_queue=None, job_queue=None, cur_epoch=0, total_examples=None,
            total_words=None, report_delay=1.0, is_corpus_file_mode=None,
    ):
        example_count, trained_word_count, raw_word_count = 0, 0, 0
        start, next_report = default_timer() - 0.00001, 1.0
        job_tally = 0
        unfinished_worker_count = self.workers

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                self._log_progress(
                    job_queue, progress_queue, cur_epoch, example_count, total_examples,
                    raw_word_count, total_words, trained_word_count, elapsed)
                next_report = elapsed + report_delay
        # all done; report the final stats
        elapsed = default_timer() - start
        self._log_epoch_end(
            cur_epoch, example_count, total_examples, raw_word_count, total_words,
            trained_word_count, elapsed, is_corpus_file_mode)
        self.total_train_time += elapsed
        return trained_word_count, raw_word_count, job_tally

    def _log_progress(
            self, job_queue, progress_queue, cur_epoch, example_count, total_examples,
            raw_word_count, total_words, trained_word_count, elapsed
    ):
        return

    def _log_epoch_end(
            self, cur_epoch, example_count, total_examples, raw_word_count, total_words,
            trained_word_count, elapsed, is_corpus_file_mode
    ):
        if is_corpus_file_mode:
            return

    def _train_epoch(
            self, data_iterable, cur_epoch=0, total_examples=None, total_words=None,
            queue_factor=2, report_delay=1.0, callbacks=(),
    ):
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [
            threading.Thread(
                target=self._worker_loop,
                args=(job_queue, progress_queue,))
            for _ in range(self.workers)
        ]

        workers.append(threading.Thread(
            target=self._job_producer,
            args=(data_iterable, job_queue),
            kwargs={'cur_epoch': cur_epoch, 'total_examples': total_examples, 'total_words': total_words}))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        trained_word_count, raw_word_count, job_tally = self._log_epoch_progress(
            progress_queue, job_queue, cur_epoch=cur_epoch, total_examples=total_examples,
            total_words=total_words, report_delay=report_delay, is_corpus_file_mode=False,
        )
        return trained_word_count, raw_word_count, job_tally

    def train(
            self, corpus_iterable=None, corpus_file=None, total_examples=None, total_words=None,
            epochs=None, start_alpha=None, end_alpha=None, word_count=0,
            queue_factor=2, report_delay=1.0, compute_loss=False, callbacks=(),
            **kwargs,
    ):
        self.alpha = start_alpha or self.alpha
        self.min_alpha = end_alpha or self.min_alpha
        self.epochs = epochs

        self._check_training_sanity(epochs=epochs, total_examples=total_examples, total_words=total_words)
        self._check_corpus_sanity(corpus_iterable=corpus_iterable, corpus_file=corpus_file, passes=epochs)

        self.add_lifecycle_event(
            "train",
            msg=(
                f"training model with {self.workers} workers on {len(self.wv)} vocabulary and "
                f"{self.layer1_size} features, using sg={self.sg} hs={self.hs} sample={self.sample} "
                f"negative={self.negative} window={self.window} shrink_windows={self.shrink_windows}"
            ),
        )

        self.compute_loss = compute_loss
        self.running_training_loss = 0.0

        for callback in callbacks:
            callback.on_train_begin(self)

        trained_word_count = 0
        raw_word_count = 0

        start = default_timer() - 0.00001
        job_tally = 0

        for cur_epoch in range(self.epochs):
            for callback in callbacks:
                callback.on_epoch_begin(self)

            if corpus_iterable is not None:
                trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch(
                    corpus_iterable, cur_epoch=cur_epoch, total_examples=total_examples,
                    total_words=total_words, queue_factor=queue_factor, report_delay=report_delay,
                    callbacks=callbacks, **kwargs)
            else:
                trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch_corpusfile(
                    corpus_file, cur_epoch=cur_epoch, total_examples=total_examples, total_words=total_words,
                    callbacks=callbacks, **kwargs)

            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch
            job_tally += job_tally_epoch

            for callback in callbacks:
                callback.on_epoch_end(self)

        # Log overall time
        total_elapsed = default_timer() - start
        self._log_train_end(raw_word_count, trained_word_count, total_elapsed, job_tally)

        self.train_count += 1  # number of times train() has been called
        self._clear_post_train()

        for callback in callbacks:
            callback.on_train_end(self)

        return trained_word_count, raw_word_count

    def _clear_post_train(self):
        self.wv.norms = None

    def _log_train_end(self, raw_word_count, trained_word_count, total_elapsed, job_tally):
        self.add_lifecycle_event("train", msg=(
            f"training on {raw_word_count} raw words ({trained_word_count} effective words) "
            f"took {total_elapsed:.1f}s, {trained_word_count / total_elapsed:.0f} effective words/s"
        ))

    def _check_corpus_sanity(self, corpus_iterable=None, corpus_file=None, passes=1):
        """Checks whether the corpus parameters make sense."""
        if corpus_file is None and corpus_iterable is None:
            raise TypeError("Either one of corpus_file or corpus_iterable value must be provided")
        if corpus_file is not None and corpus_iterable is not None:
            raise TypeError("Both corpus_file and corpus_iterable must not be provided at the same time")
        if corpus_iterable is None and not os.path.isfile(corpus_file):
            raise TypeError("Parameter corpus_file must be a valid path to a file, got %r instead" % corpus_file)
        if corpus_iterable is None:
            print("corpus_iterable is None")

    def _scan_vocab(self, sentences, progress_per, trim_rule):
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):
            if not checked_string_types:
                checked_string_types += 1
            for word in sentence:
                vocab[word] += 1
            total_words += len(sentence)

            if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                prune_vocab(vocab, min_reduce, trim_rule=trim_rule)
                min_reduce += 1

        corpus_count = sentence_no + 1
        self.raw_vocab = vocab
        return total_words, corpus_count

    def scan_vocab(self, corpus_iterable=None, corpus_file=None, progress_per=10000, workers=None, trim_rule=None):
        if corpus_file:
            corpus_iterable = LineSentence(corpus_file)
        total_words, corpus_count = self._scan_vocab(corpus_iterable, progress_per, trim_rule)
        return total_words, corpus_count

    def prepare_vocab(
            self, update=False, keep_raw_vocab=False, trim_rule=None,
            min_count=None, sample=None, dry_run=False,
    ):
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        # set effective_min_count to min_count in case max_final_vocab isn't set
        self.effective_min_count = min_count

        # If max_final_vocab is specified instead of min_count,
        # pick a min_count which satisfies max_final_vocab as well as possible.
        if self.max_final_vocab is not None:
            sorted_vocab = sorted(self.raw_vocab.keys(), key=lambda word: self.raw_vocab[word], reverse=True)
            calc_min_count = 1

            if self.max_final_vocab < len(sorted_vocab):
                calc_min_count = self.raw_vocab[sorted_vocab[self.max_final_vocab]] + 1

            self.effective_min_count = max(calc_min_count, min_count)
            self.add_lifecycle_event(
                "prepare_vocab",
                msg=(
                    f"max_final_vocab={self.max_final_vocab} and min_count={min_count} resulted "
                    f"in calc_min_count={calc_min_count}, effective_min_count={self.effective_min_count}"
                )
            )

        if not update:
            retain_total, retain_words = 0, []
            # Discard words less-frequent than min_count
            if not dry_run:
                self.wv.index_to_key = []
                # make stored settings match these applied settings
                self.min_count = min_count
                self.sample = sample
                self.wv.key_to_index = {}

            for word, v in self.raw_vocab.items():
                if keep_vocab_item(word, v, self.effective_min_count, trim_rule=trim_rule):
                    retain_words.append(word)
                    retain_total += v
                    if not dry_run:
                        self.wv.key_to_index[word] = len(self.wv.index_to_key)
                        self.wv.index_to_key.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            if not dry_run:
                # now update counts
                for word in self.wv.index_to_key:
                    self.wv.set_vecattr(word, 'count', self.raw_vocab[word])
            original_unique_total = len(retain_words) + drop_unique
            retain_unique_pct = len(retain_words) * 100 / max(original_unique_total, 1)
            self.add_lifecycle_event(
                "prepare_vocab",
                msg=(
                    f"effective_min_count={self.effective_min_count} retains {len(retain_words)} unique "
                    f"words ({retain_unique_pct:.2f}% of original {original_unique_total}, drops {drop_unique})"
                ),
            )

            original_total = retain_total + drop_total
            retain_pct = retain_total * 100 / max(original_total, 1)
            self.add_lifecycle_event(
                "prepare_vocab",
                msg=(
                    f"effective_min_count={self.effective_min_count} leaves {retain_total} word corpus "
                    f"({retain_pct:.2f}% of original {original_total}, drops {drop_total})"
                ),
            )
        else:

            new_total = pre_exist_total = 0
            new_words = []
            pre_exist_words = []
            for word, v in self.raw_vocab.items():
                if keep_vocab_item(word, v, self.effective_min_count, trim_rule=trim_rule):
                    if self.wv.has_index_for(word):
                        pre_exist_words.append(word)
                        pre_exist_total += v
                        if not dry_run:
                            pass
                    else:
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            self.wv.key_to_index[word] = len(self.wv.index_to_key)
                            self.wv.index_to_key.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            if not dry_run:
                # now update counts
                self.wv.allocate_vecattrs(attrs=['count'], types=[type(0)])
                for word in self.wv.index_to_key:
                    self.wv.set_vecattr(word, 'count', self.wv.get_vecattr(word, 'count') + self.raw_vocab.get(word, 0))
            original_unique_total = len(pre_exist_words) + len(new_words) + drop_unique
            pre_exist_unique_pct = len(pre_exist_words) * 100 / max(original_unique_total, 1)
            new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
            self.add_lifecycle_event(
                "prepare_vocab",
                msg=(
                    f"added {len(new_words)} new unique words ({new_unique_pct:.2f}% of original "
                    f"{original_unique_total}) and increased the count of {len(pre_exist_words)} "
                    f"pre-existing words ({pre_exist_unique_pct:.2f}% of original {original_unique_total})"
                ),
            )
            retain_words = new_words + pre_exist_words
            retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with higher count than sample
            threshold_count = int(sample * (3 + np.sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (np.sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.wv.set_vecattr(w, 'sample_int', np.uint32(word_probability * (2 ** 32 - 1)))

        if not dry_run and not keep_raw_vocab:
            self.raw_vocab = defaultdict(int)
        self.add_lifecycle_event(
            "prepare_vocab",
            msg=(
                f"downsampling leaves estimated {downsample_total} word corpus "
                f"({downsample_total * 100.0 / max(retain_total, 1):.1f}%% of prior {retain_total})"
            ),
        )

        # return from each step: words-affected, resulting-corpus-size, extra memory estimates
        report_values = {
            'drop_unique': drop_unique, 'retain_total': retain_total, 'downsample_unique': downsample_unique,
            'downsample_total': int(downsample_total), 'num_retained_words': len(retain_words)
        }

        if self.null_word:
            # create null pseudo-word for padding when using concatenative L1 (run-of-words)
            # this word is only ever input – never predicted – so count, huffman-point, etc doesn't matter
            self.add_null_word()

        if self.sorted_vocab and not update:
            self.wv.sort_by_descending_frequency()

        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()

        return report_values

    def add_null_word(self):
        word = '\0'
        self.wv.key_to_index[word] = len(self.wv)
        self.wv.index_to_key.append(word)
        self.wv.set_vecattr(word, 'count', 1)

    def create_binary_tree(self):
        _assign_binary_codes(self.wv)

    def _build_heap(wv):
        heap = list(Heapitem(wv.get_vecattr(i, 'count'), i, None, None) for i in range(len(wv.index_to_key)))
        heapq.heapify(heap)
        for i in range(len(wv) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(
                heap, Heapitem(count=min1.count + min2.count, index=i + len(wv), left=min1, right=min2)
            )
        return heap

    def _assign_binary_codes(wv):
        heap = _build_heap(wv)
        if not heap:
            return

        # recurse over the tree, assigning a binary code to each vocabulary word
        max_depth = 0
        stack = [(heap[0], [], [])]
        while stack:
            node, codes, points = stack.pop()
            if node[1] < len(wv):  # node[1] = index
                # leaf node => store its path from the root
                k = node[1]
                wv.set_vecattr(k, 'code', codes)
                wv.set_vecattr(k, 'point', points)
                # node.code, node.point = codes, points
                max_depth = max(len(codes), max_depth)
            else:
                # inner node => continue recursion
                points = np.array(list(points) + [node.index - len(wv)], dtype=np.uint32)
                stack.append((node.left, np.array(list(codes) + [0], dtype=np.uint8), points))
                stack.append((node.right, np.array(list(codes) + [1], dtype=np.uint8), points))

    def make_cum_table(self, domain=2 ** 31 - 1):
        vocab_size = len(self.wv.index_to_key)
        self.cum_table = np.zeros(vocab_size, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in range(vocab_size):
            count = self.wv.get_vecattr(word_index, 'count')
            train_words_pow += count ** float(self.ns_exponent)
        cumulative = 0.0
        for word_index in range(vocab_size):
            count = self.wv.get_vecattr(word_index, 'count')
            cumulative += count ** float(self.ns_exponent)
            self.cum_table[word_index] = round(cumulative / train_words_pow * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain

    def prepare_weights(self, update=False):
        # set initial input/projection and hidden weights
        if not update:
            self.init_weights()
        else:
            self.update_weights()

    def init_weights(self):
        self.wv.resize_vectors(seed=self.seed)

        if self.hs:
            self.syn1 = np.zeros((len(self.wv), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = np.zeros((len(self.wv), self.layer1_size), dtype=REAL)

    def update_weights(self):
        # Raise an error if an online update is run before initial training on a corpus
        preresize_count = len(self.wv.vectors)
        self.wv.resize_vectors(seed=self.seed)
        gained_vocab = len(self.wv.vectors) - preresize_count

        if self.hs:
            self.syn1 = np.vstack([self.syn1, np.zeros((gained_vocab, self.layer1_size), dtype=REAL)])
        if self.negative:
            pad = np.zeros((gained_vocab, self.layer1_size), dtype=REAL)
            self.syn1neg = np.vstack([self.syn1neg, pad])

    def estimate_memory(self, vocab_size=None, report=None):
        vocab_size = vocab_size or len(self.wv)
        report = report or {}
        report['vocab'] = vocab_size * (700 if self.hs else 500)
        report['vectors'] = vocab_size * self.vector_size * np.dtype(REAL).itemsize
        if self.hs:
            report['syn1'] = vocab_size * self.layer1_size * np.dtype(REAL).itemsize
        if self.negative:
            report['syn1neg'] = vocab_size * self.layer1_size * np.dtype(REAL).itemsize
        report['total'] = sum(report.values())
        return report

    def build_vocab(
            self, corpus_iterable=None, corpus_file=None, update=False, progress_per=10000,
            keep_raw_vocab=False, trim_rule=None, **kwargs,
    ):

        self._check_corpus_sanity(corpus_iterable=corpus_iterable, corpus_file=corpus_file, passes=1)
        total_words, corpus_count = self.scan_vocab(
            corpus_iterable=corpus_iterable, corpus_file=corpus_file, progress_per=progress_per,
            trim_rule=trim_rule)
        self.corpus_count = corpus_count
        self.corpus_total_words = total_words
        report_values = self.prepare_vocab(update=update, keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule,
                                           **kwargs)
        report_values['memory'] = self.estimate_memory(vocab_size=report_values['num_retained_words'])
        self.prepare_weights(update=update)
        self.add_lifecycle_event("build_vocab", update=update, trim_rule=str(trim_rule))


class Heapitem(namedtuple('Heapitem', 'count, index, left, right')):
    def __lt__(self, other):
        return self.count < other.count


def _build_heap(wv):
    heap = list(Heapitem(wv.get_vecattr(i, 'count'), i, None, None) for i in range(len(wv.index_to_key)))
    heapq.heapify(heap)
    for i in range(len(wv) - 1):
        min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(
            heap, Heapitem(count=min1.count + min2.count, index=i + len(wv), left=min1, right=min2)
        )
    return heap


class sen_get:
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        path1 = self.source
        for line in open(path1):
            yield simple_preprocess(line)


class my_layer:
    def __init__(self):
        Dimension = 300
        NumWords = 100
        WeightsInitializer = 0.01
        Weights = {}
        WeightLearnRateFactor = 1
        WeightL2Factor = 1
        Name = None
        NumInputs = 0
        InputNames = None
        NumOutputs = 1
        OutputNames = None
        wv = None

    def start(self, Dimension=300, NumWords=100, WeightsInitializer=0.01, Weights=None, WeightLearnRateFactor=1,
              WeightL2Factor=1, Name=None, NumInputs=0, InputNames=None, NumOutputs=1, OutputNames=None):
        self.Dimension = Dimension
        self.NumWords = NumWords
        self.WeightsInitializer = WeightsInitializer
        self.Weights = Weights
        self.WeightLearnRateFactor = WeightLearnRateFactor
        self.WeightL2Factor = WeightL2Factor
        self.Name = Name
        self.NumInputs = NumInputs
        self.InputNames = InputNames
        self.NumOutputs = NumOutputs
        self.OutputNames = OutputNames
        # print(self.Weights)


def wordEmbeddingLayer(dimension, numWords, **kwargs):
    a = my_layer()
    a.start(Dimension=dimension, NumWords=numWords, **kwargs)
    return a

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
# #model =wordEncoding(documents)
#
# model =wordEncoding('test_wordEncoding.txt')
# model =wordEncoding('test_wordEncoding.txt',Dimension=100)
# model =wordEncoding('test_wordEncoding.txt',Window=10)
# model =wordEncoding('test_wordEncoding.txt',MinCount=8,InitialLearnRate=0.05)
#
# layer =wordEmbeddingLayer(100,8,Weights=model)
#
# print(model)
# vec_king = word2vec(model,'ten')
# print(vec_king)
# print(model.index_to_key)
# print('111')
# dictionary2=wordEncoding(documents)
# dictionary1=wordEncoding('test_wordEncoding.txt')
# print(dictionary1)
# print(dictionary1.token2id)
#
# new_doc=['thou art bud never art art','art art bud bud']
#
# new_vec = doc2sequence(dictionary1,new_doc)
# print(new_vec)
# wv=fastTextWordEmbedding()
# new_doc=['never emperor of the','the the of king queen']
# new_vec=doc2sequence(wv,new_doc)
# print(new_vec)
# new_vec=doc2sequence(wv,new_doc,'PaddingDirection','left')
# print(new_vec)
# new_doc=['never emperor of the','the the of king queen']
# new_vec=doc2sequence(wv,new_doc,'PaddingDirection','right','PaddingValue',66)
# print(new_vec)

# texts = [
#     [word for word in document.lower().split()]
#     for document in documents
# ]
# a = wordEncoding(texts)
# b=wordEncoding("test_wordEncoding.txt")
# print(a.token2id)
# print(a)


#
# wv=fastTextWordEmbedding()
# writeWordEmbedding(wv,"my_model2.bin")
#
# print(isVocabularyWord(wv,'of'))
# print(isVocabularyWord(wv,['of']))
# print(isVocabularyWord(wv,['of','the','emperor']))
# print(isVocabularyWord(wv,['queen','the','king']))
#
#
# print(word2ind(wv,'of'))
# print(word2ind(wv,['of']))
# print(word2ind(wv,['of','the','emperor']))
# print(word2ind(wv,['queen','the','king']))
#
# print(ind2word(wv,0))
# print(ind2word(wv,[0,1,2]))
# print(ind2word(wv,[3,6,8]))
#
# print(wv.key_to_index)
# print(wv.index_to_key)
# print(wv.vectors)
#
#
#
# vec_king=word2vec(wv,"king")
# vec_woman=word2vec(wv,"woman")
# vec_man=word2vec(wv,"man")
#
# vec_ans=vec_king-vec_man+vec_woman
# my_ans=vec2word(wv,vec_king+vec_woman-vec_man)
#
# print(my_ans)
