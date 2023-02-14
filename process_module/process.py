import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from utils import preprocess_en_doc, preprocess_zh_doc


def process(args):
    """
        data_name : dataset name
        vocabulary_size: vocabulary size
        input_path: data path
        input_file: data name. Data must include 'text', 'title' columns. Could have 'title'
                    'author', 'journal', 'date' columns. More than one authors should
                    be seperated by ';'
        lang: language
        output_path: output path
    """
    ### create output file destination
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    ### load data
    data = pd.read_excel(args.input_path + args.input_file)
    ## save the document
    document_columns = []
    assert "title" in list(data.columns), "data format error,there is no title column in this corpus"
    assert "text" in list(data.columns), "data format error,there is no text column in this corpus"
    document_columns.append("title")
    document_columns.append("text")
    if "date" in data.columns:
        document_columns.append("date")
    document_entity = data[document_columns]
    document_entity["document_id"] = range(len(document_entity))
    document_entity.to_csv(f"{args.output_path}document.csv", index=False)

    ### clean the document
    if args.lang == "en":
        data["text"] = data["text"].apply(preprocess_en_doc)
    else:
        data["text"] = data["text"].apply(preprocess_zh_doc)
    corpus = data["text"].values
    ########## bulid vocabulary #############
    vocabulary_size = args.vocabulary_size
    cv = CountVectorizer(
        strip_accents='unicode',  # 将使用unicode编码在预处理步骤去除raw document中的重音符号
        max_features=args.vocabulary_size,
        max_df=0.5,  # 阈值如果某个词的document frequency大于max_df，不当作关键词
        min_df=10  # 如果某个词的document frequency小于min_df，则这个词不会被当作关键词
    )
    tf = cv.fit_transform(corpus)
    #### save the vocabulary and bow
    cv_path = 'cv.pkl'
    bow_path = "bow.pkl"
    with open(f"{args.output_path}{cv_path}", 'wb') as fw:
        pickle.dump(cv.vocabulary_, fw)
    with open(f"{args.output_path}{bow_path}", 'wb') as fw:
        pickle.dump(tf, fw)
    ######### save the entity ###########
    word2idx, idx2word, author2idx, idx2author, journal2idx, idx2journal = None, None, None, None, None, None
    #### save document`s relationship between other entity
    ### journal 
    if "journal" in list(data.columns):
        ## 建立的journal 的实体
        journal_list = data["journal"].apply(lambda x: x.strip()).unique()
        idx2journal = dict([(key, val) for key, val in zip(range(len(journal_list)), list(journal_list))])
        journal2idx = dict([(val, key) for key, val in zip(range(len(journal_list)), list(journal_list))])
        journal_entity = pd.DataFrame(
            data=idx2journal.items(),
            columns=["journal_id", "journal_name"]
        )
        journal_entity.to_csv(f"{args.output_path}journal.csv", index=False)
        ## 建立journal 与document 的关系
        document_journal_relation = []
        for document_idx, journal in enumerate(data["journal"].values):
            document_journal_relation.append({
                "document_id": document_idx,
                "journal_id": journal2idx[journal]
            })
        docuemnt_journal_df = pd.DataFrame(data=document_journal_relation, columns=["document_id", "journal_id"])
        docuemnt_journal_df.to_csv(f"{args.output_path}document_journal.csv", index=False)
    ### author
    if "authors" in list(data.columns):
        author_set = set()
        author_list = [authors.split(";") for authors in data["authors"].values]
        for ll in author_list:
            for author in ll:
                author_set.add(author.strip())
        if " " in list(author_set):
            author_set.discard(" ")
        if "" in list(author_set):
            author_set.discard("")
        idx2author = dict([(key, val) for key, val in zip(range(len(list(author_set))), list(author_set))])
        author2idx = dict([(val, key) for key, val in zip(range(len(list(author_set))), list(author_set))])
        author_entity = pd.DataFrame(data=idx2author.items(), columns=["author_id", "author_name"])
        author_entity.to_csv(f"{args.output_path}author.csv", index=False)
        #### the relationship between author and document
        #### document map 作者
        document_author_relation = []
        for document_idx, authors in enumerate(data["authors"].values):
            for author in authors.split(";"):
                if author.strip() in author2idx.keys():
                    document_author_relation.append(
                        {
                            "document_id": document_idx,
                            "author_id": author2idx[author.strip()],
                        }
                    )
        docuemnt_author_df = pd.DataFrame(data=document_author_relation, columns=["document_id", "author_id"])
        docuemnt_author_df.to_csv(f"{args.output_path}document_author.csv", index=False)
    ### word 实体建立
    vocab = cv.get_feature_names_out()
    freq = tf.sum(axis=0).getA1()
    word2idx = dict([(word, idx) for idx, word in enumerate(vocab)])
    idx2word = dict([(idx, word) for idx, word in enumerate(vocab)])
    # word2idx = cv.vocabulary_
    # idx2word = dict([(idx, word) for word, idx in cv.vocabulary_.items()])
    # word_entity = pd.DataFrame(data=idx2word.items(), columns=["word_id", "word_key"])
    word_entity = pd.DataFrame(data={"word_id": range(len(vocab)), "word_key": vocab, "word_frequency": freq})
    word_entity.to_csv(f"{args.output_path}word.csv", index=False)
    #### document 与word 的共现关系
    document_word_occurrence = tf.toarray()
    document_word_occurrence[document_word_occurrence < 10] = 0
    document_word_relation = []
    for document_idx in range(len(data)):
        for word_idx, occur_num in enumerate(document_word_occurrence[document_idx, :]):
            if occur_num:
                document_word_relation.append(
                    {
                        "document_id": document_idx,
                        "word_id": word_idx,
                        "frequency": occur_num,
                    }
                )
    docuemnt_word_df = pd.DataFrame(data=document_word_relation,
                                    columns=["document_id", "word_id", "frequency"])  # 3037434
    docuemnt_word_df.to_csv(f"{args.output_path}/document_word.csv", index=False)
    #### journal map 作者
    if len(list({"authors", "journal"} & set(data.columns))) == 2:
        authors_journal_relation = []
        for document_idx, authors_journal_pair in enumerate(data[["authors", "journal"]].values):
            authors, journal = authors_journal_pair
            authors = [author.strip() for author in authors.split(";")]
            for author in authors:
                if author in author2idx.keys():
                    authors_journal_relation.append(
                        {
                            "document_id": document_idx,
                            "journal_id": journal2idx[journal],
                            "author_id": author2idx[author]
                        }
                    )
        authors_journal_df = pd.DataFrame(data=authors_journal_relation,
                                          columns=["document_id", "author_id", "journal_id"])
        # authors_journal_df = authors_journal_df.groupby(["author_id","journal_id"]).agg([np.sum])
        authors_journal_df.to_csv(f"{args.output_path}authors_journal.csv", index=False)
    ### word_map_word : word与word 同时在文章里出现的次数
    word_word_co_occurence = np.zeros((vocabulary_size, vocabulary_size))
    for document_idx in range(len(data)):
        x = document_word_occurrence[document_idx, :]
        occur_idx = np.argwhere(x > 0).reshape(-1)
        for i in range(len(occur_idx)):
            for j in range(i + 1, len(occur_idx)):
                word_word_co_occurence[occur_idx[i]][occur_idx[j]] += min(x[occur_idx[i]], x[occur_idx[j]])
    word_word_occur_relationship = []
    for i in range(vocabulary_size):
        for j in range(i + 1, vocabulary_size):
            if word_word_co_occurence[i][j]:
                word_word_occur_relationship.append(
                    {
                        "word_id": i,
                        "aim_word_id": j,
                        "value": word_word_co_occurence[i][j],  # co_occurence
                        "relationship_type": 1
                    }
                )
    word_word_df = pd.DataFrame(data=word_word_occur_relationship,
                                columns=["word_id", "aim_word_id", "value", "relationship_type"])
    word_word_df.to_csv(f"{args.output_path}word_word.csv", index=False)
    return (word2idx, idx2word), \
           (author2idx, idx2author), \
           (journal2idx, idx2journal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Topic Model data process')
    # parser.add_argument('--data_name', type=str, required=True)
    # parser.add_argument('--input_path', type=str, required=True)
    # parser.add_argument('--input_file', type=str, required=True)
    # parser.add_argument('--lang', type=str, required=True)
    # parser.add_argu∏ment('--output_path', type=str,required=True)
    parser.add_argument('--data_name', type=str, default="test")
    parser.add_argument('--vocabulary_size', type=int, default=2 ** 12)
    parser.add_argument('--input_path', type=str, default="./dataset/test-500/")
    parser.add_argument('--input_file', type=str, default="data.xlsx")
    parser.add_argument('--lang', type=str, default="en")
    parser.add_argument('--output_path', type=str, default="./output/test-500-gsm/")
    #### load config
    args = parser.parse_args()
    word_dict, author_dict, journal_dict = process(args)
    print("process done")
