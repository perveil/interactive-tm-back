import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import argparse
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import warnings
warnings.filterwarnings("ignore")
from models.ETM.ETM_model import ETM
from models.utils.fn import DocDataset
import torch
from models.utils.fn import _get_vocab, _get_doc_lengths, _get_term_freqs, _row_norm, _build_entity


def run_etm(args):
    """
       n_components : topic_num
       max_iter: max iteration for model training
       embed_dim: dimension of topic vector
       batch_size: batch size for training
       lr: learning rate
       use_gpu: use gpu to train or not
       ckpt: file path of last training checkpoint
       log_every: how many epochs to save a checkpoint
    """
    cv = CountVectorizer(vocabulary=pickle.load(open(f"{args.input_path}/cv.pkl", 'rb')))
    bow = pickle.load(open(f"{args.input_path}/bow.pkl", 'rb'))

    vocab = _get_vocab(cv)
    doc_lengths = _get_doc_lengths(bow)
    term_freqs = _get_term_freqs(bow)

    doc_dataset = DocDataset(vocab, bow, doc_lengths, term_freqs)
    print(f"ETM training begins.\n"
          f"Topic num: {args.n_components}. Total iters: {args.max_iter}\n"
          f"Vocab size: {len(vocab)}, Doc size: {len(doc_lengths)}")
    model = ETM(
        vocab=vocab,
        bow_dim=len(vocab),
        n_topic=args.n_components,
        task_name=args.data_name,
        device="cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu",
        emb_dim=args.embed_dim
    )
    ckpt = None if args.ckpt == "" else torch.load(args.ckpt)
    model.train(
        train_data=doc_dataset,
        test_data=doc_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.max_iter,
        log_every=args.log_every,
        beta=1.0,
        ckpt=ckpt,
    )
    topic_word_dist = model.get_topic_word_dist()
    topic_word_dist = _row_norm(topic_word_dist)
    print(len(topic_word_dist), len(topic_word_dist[0]))
    print(topic_word_dist)
    doc_topic_dist = model.get_doc_topic_dist(doc_dataset)
    doc_topic_dist = _row_norm(doc_topic_dist)
    print(len(doc_topic_dist), len(doc_topic_dist[0]))
    print(doc_topic_dist)
    return topic_word_dist, doc_topic_dist, vocab, doc_lengths, term_freqs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Topic Model ETM')
    parser.add_argument('--data_name', type=str, default="covid-2022-11")
    parser.add_argument('--input_path', type=str, default="./dataset/covid-2022-11-output/")
    parser.add_argument('--output_path', type=str, default="./dataset/covid-2022-11-output/")
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--log_every', type=int, default=20)
    #### load config
    args = parser.parse_args()
    topic_word_dis, doc_topic_dis, vocab, doc_lengths, term_freqs = run_etm(args)
    print("ETM training End")
    _build_entity(args, topic_word_dis, doc_topic_dis, vocab, doc_lengths, term_freqs)
    print("ETM End")