import numpy as np
import pandas as pd
from gensim.corpora import Dictionary

from utils import create_topic_entity
from sklearn.metrics.pairwise import cosine_similarity


def _get_doc_lengths(dtm):
    return dtm.sum(axis=1).getA1()


def _row_norm(dists):
    # row normalization function required
    # for doc_topic_dists and topic_term_dists
    return dists / dists.sum(axis=1)[:, None]


def _get_term_freqs(dtm):
    return dtm.sum(axis=0).getA1()


def _get_vocab(vectorizer):
    return vectorizer.get_feature_names_out()
    
def _build_entity(args, topic_word_dis, doc_topic_dis, vocab, doc_lengths, term_freqs):
    ### create output file destination
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    #### build topic entity
    topic_entity = create_topic_entity(topic_word_dis, doc_topic_dis, vocab, doc_lengths, term_freqs, start_index=0)
    topic_entity.to_csv(f"{args.output_path}topic.csv", index=False)
    ### build word_word semantic relationship
    ### find top10 the most similar  semantic use cosine_similarity
    N, V = topic_word_dis.shape
    word_word_relationship = []
    word_word_semantic_similarity = cosine_similarity(topic_word_dis.transpose(1, 0))
    word_word_semantic_similarity[
        np.arange(V)[:, None],
        np.argpartition(word_word_semantic_similarity, V - 10, axis=-1)[:, :-10]
    ] = 0
    np.fill_diagonal(word_word_semantic_similarity, 0)
    index_x, index_y = np.where(word_word_semantic_similarity > 0)
    for x, y in zip(index_x.tolist(), index_y.tolist()):
        word_word_relationship.append(
            {
                "word_id": x,
                "aim_word_id": y,
                "value": word_word_semantic_similarity[x][y],  # similarity value
                "relationship_type": 2
            }
        )
    word_word_df_2 = pd.DataFrame(data=word_word_relationship,
                                  columns=["word_id", "aim_word_id", "value", "relationship_type"])
    word_word_df_1 = pd.read_csv(f"{args.output_path}word_word.csv", index_col=None)
    word_word_df = pd.concat([word_word_df_2, word_word_df_1], axis=0)
    word_word_df.to_csv(f"{args.output_path}word_word.csv", index=False)
    ### build topic distribution for document
    topic_word_relationship = []
    for topic_id in range(args.n_components):
        for word_id in range(len(vocab)):
            topic_word_relationship.append({
                "topic_id": topic_id,
                "word_id": word_id,
                "distribution_value": topic_word_dis[topic_id][word_id],
                "relationship_type": 1  ## distribution maybe seedword
            }
        )
    topic_word_df = pd.DataFrame(data=topic_word_relationship, columns=["topic_id", "word_id", "distribution_value"])
    topic_word_df.to_csv(f"{args.output_path}topic_word.csv", index=False)
    ### build topic distribution for word
    topic_document_relationship = []
    for topic_id in range(args.n_components):
        for docu_id in range(len(doc_lengths)):
            topic_document_relationship.append({
                "topic_id": topic_id,
                "document_id": docu_id,
                "distribution_value": doc_topic_dis[docu_id][topic_id]
            }
        )
    topic_document_df = pd.DataFrame(data=topic_document_relationship,
                                     columns=["topic_id", "document_id", "distribution_value"])
    topic_document_df.to_csv(f"{args.output_path}topic_document.csv", index=False)

    ### build authors' documents' topic distribution
    author_entity = pd.read_csv(f"{args.output_path}author.csv")
    document_author_df = pd.read_csv(f"{args.output_path}document_author.csv")
    author_document_topic_relationship = []

    def get_adt_relationship(x):
        author_id = x.iloc[0, 1]
        topic_frequency = {}
        for document_id in x.document_id:
            current_document_topic = topic_document_df[topic_document_df['document_id'] == document_id]
            max_distribution_rid = current_document_topic.distribution_value.argmax()
            topic_id = current_document_topic.iloc[max_distribution_rid, 0]
            if topic_id in topic_frequency.keys():
                topic_frequency[topic_id] += 1
            else:
                topic_frequency[topic_id] = 1
        for topic_id, frequency in topic_frequency.items():
            author_document_topic_relationship.append({
                'author_id': author_id,
                'topic_id': topic_id,
                'frequency': frequency,
            })

    document_author_df.groupby('author_id').apply(get_adt_relationship)
    author_document_topic_df = pd.DataFrame(data=author_document_topic_relationship,
                                            columns=["author_id", "topic_id", "frequency"])
    author_document_topic_df.to_csv(f"{args.output_path}author_topic_frequency.csv", index=False)


#### model evaluation
import os
import gensim
import numpy as np
from gensim.models.coherencemodel import CoherenceModel

def get_topic_words(model,topn=15,n_topic=10,vocab=None,fix_topic=None,showWght=False):
    topics = []
    def show_one_tp(tp_idx):
        if showWght:
            return [(vocab.id2token[t[0]],t[1]) for t in model.get_topic_terms(tp_idx,topn=topn)]
        else:
            return [vocab.id2token[t[0]] for t in model.get_topic_terms(tp_idx,topn=topn)]
    if fix_topic is None:
        for i in range(n_topic):
            topics.append(show_one_tp(i))
    else:
        topics.append(show_one_tp(fix_topic))
    return topics


def calc_topic_diversity(topic_words):
    """
        topic_words: topic words in the form of [[w11,w12,...],[w21,w22,...]]
    """
    vocab = set(sum(topic_words, []))
    n_total = len(topic_words) * len(topic_words[0])
    topic_div = len(vocab) / n_total
    return topic_div


def calc_topic_coherence(topic_words,docs,dictionary,emb_path=None,taskname=None,sents4emb=None,calc4each=False):
    """
        topic_words: topic words in the form of [[w11,w12,...],[w21,w22,...]]
        docs: documents in the form of [[w11,w12,...],[w21,w22,...]], i.e. list of list of str, tokenized texts
        dictionary: Gensim dictionary mapping of id word
        emb_path: path of the pretrained word2vec weights in text format, used in calculating c_w2v score
        sents4emb: list/generator of tokenized sentences, used to train embeddings in calculating c_w2v score
        calc4each: whether to calculate each topic's coherence
    """
    # Computing the C_V score
    cv_coherence_model = CoherenceModel(topics=topic_words, texts=docs, dictionary=dictionary, coherence='c_v',
                                        processes=2)
    cv_per_topic = cv_coherence_model.get_coherence_per_topic() if calc4each else None
    cv_score = cv_coherence_model.get_coherence()
    
    # Computing the C_W2V score
    try:
        w2v_model_path = f'./models/ETM/ckpt/{taskname}_w2v_weight.kv'
        # Priority order: 1) user's embed file; 2) standard path embed file; 3) train from scratch then store.
        if emb_path is not None and os.path.exists(emb_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=False)
        elif os.path.exists(w2v_model_path):
            keyed_vectors = gensim.models.KeyedVectors.load(w2v_model_path)
        elif sents4emb is not None:
            print('Training a word2vec model 20 epochs to evaluate topic coherence, this may take a few minutes ...')
            w2v_model = gensim.models.Word2Vec(sents4emb, min_count=1, workers=6)
            keyed_vectors = w2v_model.wv
            keyed_vectors.save(w2v_model_path)
        else:
            raise Exception("C_w2v score isn't available for the missing of training corpus (sents4emb=None).")

        w2v_coherence_model = CoherenceModel(topics=topic_words, texts=docs, dictionary=dictionary, coherence='c_w2v',
                                             keyed_vectors=keyed_vectors, processes=2)
        w2v_per_topic = w2v_coherence_model.get_coherence_per_topic() if calc4each else None
        w2v_score = w2v_coherence_model.get_coherence()
    except Exception as e:
        print(e)
        w2v_per_topic = [None for _ in range(len(topic_words))]
        w2v_score = None

    # Computing the C_UCI score
    c_uci_coherence_model = CoherenceModel(topics=topic_words, texts=docs, dictionary=dictionary, coherence='c_uci',
                                           processes=2)
    c_uci_per_topic = c_uci_coherence_model.get_coherence_per_topic() if calc4each else None
    c_uci_score = c_uci_coherence_model.get_coherence()

    # Computing the C_NPMI score
    c_npmi_coherence_model = CoherenceModel(topics=topic_words, texts=docs, dictionary=dictionary, coherence='c_npmi',
                                            processes=2)
    c_npmi_per_topic = c_npmi_coherence_model.get_coherence_per_topic() if calc4each else None
    c_npmi_score = c_npmi_coherence_model.get_coherence()

    # Computing the C_Mimno score
    c_mimno_score = mimno_topic_coherence(topic_words, docs)
    return (cv_score, w2v_score, c_uci_score, c_npmi_score, c_mimno_score), \
           (cv_per_topic, w2v_per_topic, c_uci_per_topic, c_npmi_per_topic)


def mimno_topic_coherence(topic_words, docs):
    tword_set = set([w for wlst in topic_words for w in wlst])
    word2docs = {w:set([]) for w in tword_set}
    for docid,doc in enumerate(docs):
        doc = set(doc)
        for word in tword_set:
            if word in doc:
                word2docs[word].add(docid)
    def co_occur(w1,w2):
        return len(word2docs[w1].intersection(word2docs[w2]))+1
    scores = []
    for wlst in topic_words:
        s = 0
        for i in range(1,len(wlst)):
            for j in range(0,i):
                s += np.log((co_occur(wlst[i],wlst[j])+1.0)/len(word2docs[wlst[j]]))
        scores.append(s)
    return np.mean(scores)


def evaluate_topic_quality(topic_words, test_data, taskname=None, calc4each=False):
    """
        topic_words: topic words in the form of [[w11,w12,...],[w21,w22,...]]
        test_data: documents in the form of [[w11,w12,...],[w21,w22,...]], i.e. list of list of str, tokenized texts
        calc4each: whether to calculate each topic's coherence
    """
    # calculate topic diversity
    topic_diversity = calc_topic_diversity(topic_words)
    print(f'topic diversity:{topic_diversity}')
    # calculate topic coherence
    dictionary = Dictionary(test_data)
    (c_v, c_w2v, c_uci, c_npmi, c_mimno),\
        (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic) = \
        calc_topic_coherence(topic_words=topic_words, docs=test_data, dictionary=dictionary,
                             emb_path=None, taskname=taskname, sents4emb=test_data, calc4each=calc4each)
    print('c_v:{}, c_w2v:{}, c_uci:{}, c_npmi:{}'.format(
        c_v, c_w2v, c_uci, c_npmi))
    scrs = {'c_v': cv_per_topic, 'c_w2v': c_w2v_per_topic, 'c_uci': c_uci_per_topic, 'c_npmi': c_npmi_per_topic}
    if calc4each:
        for scr_name, scr_per_topic in scrs.items():
            print(f'{scr_name}:')
            for t_idx, (score, twords) in enumerate(zip(scr_per_topic, topic_words)):
                print(f'topic.{t_idx + 1:>03d}: {score} {twords}')

    print('mimno topic coherence:{}'.format(c_mimno))
    if calc4each:
        return (c_v, c_w2v, c_uci, c_npmi, c_mimno, topic_diversity), (cv_per_topic, c_w2v_per_topic, c_uci_per_topic, c_npmi_per_topic)
    else:
        return c_v, c_w2v, c_uci, c_npmi, c_mimno, topic_diversity


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for pt in points:
        if smoothed_points:
            prev = smoothed_points[-1]
            smoothed_points.append(prev*factor+pt*(1-factor))
        else:
            smoothed_points.append(pt)
    return smoothed_points

#### dataset for a format

import torch
from torch.utils.data import Dataset
class DocDataset(Dataset):
    def __init__(self, vocab, bow, doc_lengths, term_freqs):
        self.vocab = vocab              # 词典，['word1', 'word2', ...]
        self.vocab_size = len(vocab)
        self.bow = bow
        self.doc_lengths = doc_lengths  # 每篇文章的长度
        self.term_freqs = term_freqs    # 每个词的总出现频率

    def __getitem__(self, idx):
        bow_vec = torch.tensor(self.bow.toarray()[idx]).float()
        return bow_vec  # tensor[freq_of_word1, freq_of_word2, ...]

    def __len__(self):
        return len(self.doc_lengths)