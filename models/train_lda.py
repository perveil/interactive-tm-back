import os
import json
import argparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import warnings

warnings.filterwarnings("ignore")
from models.model_utils import _get_vocab, _get_doc_lengths, _get_term_freqs, _row_norm, _build_entity


def run_lda(args):
    """
       n_components : topic_num
       max_iter: max iteration for model training
       priori_topic_word_dis: float, optional (default=None)
            Prior of topic word distribution `beta`. If the value is None, defaults
            to `1 / n_components`.
            In [1]_, this is called `eta`.
       priori_topic_doc_dis:  float, optional (default=None)
            Prior of document topic distribution `theta`. If the value is None,
            defaults to `1 / n_components`.
            In [1]_, this is called `alpha`.
    """
    cv = CountVectorizer(vocabulary=pickle.load(open(f"{args.input_path}/cv.pkl", 'rb')))
    bow = pickle.load(open(f"{args.input_path}/bow.pkl", 'rb'))
    priori_topic_word_dis, priori_topic_doc_dis = None, None
    if args.priori_topic_doc_dis:
        assert args.priori_topic_doc_dis.shape == (bow.toarray().shape[0], args.n_components)
        priori_topic_doc_dis = None
    if args.priori_topic_word_dis:
        assert args.priori_topic_doc_dis.shape == (args.n_components, len(cv.vocabulary_))
        priori_topic_word_dis = None
    print(f"Topic num: {args.n_components}. Total iters: {args.max_iter}")
    lda = LatentDirichletAllocation(
        n_components=args.n_components,
        max_iter=args.max_iter,
        learning_method='online',
        learning_offset=50.,
        random_state=0,
        topic_word_prior=priori_topic_word_dis,
        doc_topic_prior=priori_topic_doc_dis
    )
    ldamodel = lda.fit_transform(bow)
    #### output  entity_num * n_components

    vocab = _get_vocab(cv)
    doc_lengths = _get_doc_lengths(bow)
    term_freqs = _get_term_freqs(bow)

    return _row_norm(lda.components_), _row_norm(ldamodel), vocab, doc_lengths, term_freqs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wind Power Forecasting')
    # parser.add_argument('--data_name', type=str, required=True)
    # parser.add_argument('--input_path', type=str, required=True)
    # parser.add_argument('--output_path', type=str,required=True)
    # parser.add_argument('--priori_topic_word_dis', type=str)
    # parser.add_argument('--priori_topic_doc_dis', type=str)
    # parser.add_argument('--n_components', type=int,default=5)
    # parser.add_argument('--max_iter', type=int,default=100)
    parser.add_argument('--data_name', type=str, default="covid-2022-11")
    parser.add_argument('--input_path', type=str, default="./dataset/covid-2022-11-output/")
    parser.add_argument('--output_path', type=str, default="./dataset/covid-2022-11-output/")
    parser.add_argument('--priori_topic_word_dis', type=str)
    parser.add_argument('--priori_topic_doc_dis', type=str)
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=5)
    #### load config
    args = parser.parse_args()
    topic_word_dis, doc_topic_dis, vocab, doc_lengths, term_freqs = run_lda(args)
    print("Train done.")
    _build_entity(args, topic_word_dis, doc_topic_dis, vocab, doc_lengths, term_freqs)
    print("LDA training End")
