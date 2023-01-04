import numpy as np
import pandas as pd
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
    return vectorizer.get_feature_names()


def _build_entity(args, topic_word_dis, doc_topic_dis, vocab, doc_lengths, term_freqs):
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
