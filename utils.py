import jieba
import jieba.posseg as psg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import  string
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform


#### load english stop word
EN_STOP_WORDS = []
with open('./process_module/process_helper/en_stopwords.txt',encoding='utf-8') as file:
    for line in file.readlines():
        EN_STOP_WORDS.append(line.strip("\n"))
#### load chinese stop word
ZH_STOP_WORDS = []
with open('./process_module/process_helper/zh_stopwords.txt',encoding='utf-8') as file:
    for line in file.readlines():
        ZH_STOP_WORDS.append(line.strip("\n"))


def contains_punctuation(w):
    return any(char in string.punctuation for char in w)

def contains_numeric(w):
    return any(char.isdigit() for char in w)

def preprocess_en_doc(sentence):
    sentence = sentence.replace("\n"," ")
    in_docs = sentence.split()
    in_docs = [w.lower() for w in in_docs if not contains_punctuation(w)]  ### 是否有符号
    in_docs = [w for w in in_docs if  w not in  EN_STOP_WORDS] ### remove the stopwords
    in_docs = [w for w in in_docs if not contains_numeric(w) and len(w)>1] ### remove the str that contain char
    return " ".join(in_docs)

def preprocess_zh_doc(sentence):
    sentence = sentence.replace("\n"," ")
    res = []
    checkarr = ['n']
    for word,flag in psg.lcut(sentence):
        if (flag in checkarr) and (word not in ZH_STOP_WORDS) and (len(word)>1):
            res.append(word)
    return " ".join(res)



def softmax(x):
    
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def _jensen_shannon(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def _pcoa(pair_dists, n_components=2):
    """Principal Coordinate Analysis,
    aka Classical Multidimensional Scaling
    """
    # code referenced from skbio.stats.ordination.pcoa
    # https://github.com/biocore/scikit-bio/blob/0.5.0/skbio/stats/ordination/_principal_coordinate_analysis.py

    # pairwise distance matrix is assumed symmetric
    pair_dists = np.asarray(pair_dists, np.float64)

    # perform SVD on double centred distance matrix
    n = pair_dists.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = - H.dot(pair_dists ** 2).dot(H) / 2
    eigvals, eigvecs = np.linalg.eig(B)

    # Take first n_components of eigenvalues and eigenvectors
    # sorted in decreasing order
    ix = eigvals.argsort()[::-1][:n_components]
    eigvals = eigvals[ix]
    eigvecs = eigvecs[:, ix]

    # replace any remaining negative eigenvalues and associated eigenvectors with zeroes
    # at least 1 eigenvalue must be zero
    eigvals[np.isclose(eigvals, 0)] = 0
    if np.any(eigvals < 0):
        ix_neg = eigvals < 0
        eigvals[ix_neg] = np.zeros(eigvals[ix_neg].shape)
        eigvecs[:, ix_neg] = np.zeros(eigvecs[:, ix_neg].shape)

    return np.sqrt(eigvals) * eigvecs
def js_PCoA(distributions):
    """Dimension reduction via Jensen-Shannon Divergence & Principal Coordinate Analysis
    (aka Classical Multidimensional Scaling)
    Parameters
    ----------
    distributions : array-like, shape (`n_dists`, `k`)
        Matrix of distributions probabilities.
    Returns
    -------
    pcoa : array, shape (`n_dists`, 2)
    """
    dist_matrix = squareform(pdist(distributions, metric=_jensen_shannon))
    return _pcoa(dist_matrix)

def _topic_coordinates(mds, topic_term_dists, topic_proportion, start_index=1):
    K = topic_term_dists.shape[0]
    mds_res = mds(topic_term_dists)
    assert mds_res.shape == (K, 2)
    mds_df = pd.DataFrame({'x': mds_res[:, 0], 'y': mds_res[:, 1],
                           'topic_id': range(start_index, K + start_index),
                           "topic_name": ["topic_"+str(i) for i in range(start_index, K + start_index)],
                           'Frequency': topic_proportion * 100
                           }
                          )
    # note: cluster (should?) be deprecated soon. See: https://github.com/cpsievert/LDAvis/issues/26
    return mds_df
def _df_with_names(data, index_name, columns_name):
    if type(data) == pd.DataFrame:
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        df = pd.DataFrame(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df


def _series_with_name(data, name):
    if type(data) == pd.Series:
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)

### create a topic entity
def create_topic_entity(topic_term_dists,doc_topic_dists, vocab,doc_lengths, term_frequency,
            R=30, mds=js_PCoA, start_index=1
    ):

    topic_term_dist_cols = [
        pd.Series(topic_term_dist, dtype="float64")
        for topic_term_dist in topic_term_dists
    ]
    topic_term_dists = pd.concat(topic_term_dist_cols, axis=1).T

    topic_term_dists = _df_with_names(topic_term_dists, 'topic', 'term')
    doc_topic_dists = _df_with_names(doc_topic_dists, 'doc', 'topic')
    term_frequency = _series_with_name(term_frequency, 'term_frequency')
    doc_lengths = _series_with_name(doc_lengths, 'doc_length')
    vocab = _series_with_name(vocab, 'vocab')
    R = min(R, len(vocab))

    topic_freq = doc_topic_dists.mul(doc_lengths, axis="index").sum()
    topic_proportion = (topic_freq / topic_freq.sum())
    topic_order = topic_proportion.index
    # reorder all data based on new ordering of topics
    topic_freq = topic_freq[topic_order]
    topic_term_dists = topic_term_dists.iloc[topic_order]

    term_topic_freq = (topic_term_dists.T * topic_freq).T

    term_frequency = np.sum(term_topic_freq, axis=0)

    topic_entity = _topic_coordinates(mds, topic_term_dists, topic_proportion, start_index)

    return topic_entity

def convertStr2Dict(s):
    fn_parts = s.split('.')
    obj = {}
    keys = fn_parts[::2]
    values = fn_parts[1::2]
    for idx,key in enumerate(keys):
        obj[key] = values[idx]
    return obj

def convertStr2arg(s):
    fn_parts = s.split('.')
    res = []
    keys = fn_parts[::2]
    values = fn_parts[1::2]
    for key,value in zip(keys,values):
        res.append("--"+key+" "+value)
    return res