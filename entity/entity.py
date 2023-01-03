# coding: utf-8
from sqlalchemy import BigInteger, Column, Date, DateTime, Integer, String, text
from sqlalchemy.dialects.mysql import TEXT, VARCHAR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
metadata = Base.metadata


class Author(Base):
    __tablename__ = 'author'

    id = Column(BigInteger, primary_key=True)
    author_id = Column(BigInteger)
    author_name = Column(VARCHAR(255))
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)
    add_time = Column(VARCHAR(255))


class DMapT(Base):
    __tablename__ = 'd_map_t'

    t_name = Column(VARCHAR(255))
    value = Column(VARCHAR(255))
    topic_id = Column(VARCHAR(225))
    document_id = Column(BigInteger)
    add_time = Column(VARCHAR(255))
    id = Column(BigInteger, primary_key=True)
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)


class DMapW(Base):
    __tablename__ = 'd_map_w'

    value = Column(Integer)
    word_key = Column(VARCHAR(255))
    word_id = Column(BigInteger)
    document_id = Column(BigInteger)
    add_time = Column(VARCHAR(255))
    id = Column(BigInteger, primary_key=True)
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)


class Dataset(Base):
    __tablename__ = 'dataset'

    dataset_id = Column(BigInteger, primary_key=True)
    name = Column(VARCHAR(255), comment='文章名')
    time = Column(VARCHAR(255), comment='时间')
    institute = Column(VARCHAR(255), comment='机构')
    language = Column(VARCHAR(255), comment='语言')
    document_num = Column(BigInteger, comment='文章数')
    details = Column(String(45, 'utf8mb4_general_ci'))
    input_path = Column(VARCHAR(225))
    output_path = Column(VARCHAR(225))
    is_seed_word = Column(Integer, server_default=text("'0'"), comment='是否训练')
    is_word_embedding = Column(Integer, server_default=text("'0'"))
    is_document_label = Column(Integer, server_default=text("'0'"))
    is_document_link = Column(Integer, server_default=text("'0'"))
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')


class Document(Base):
    __tablename__ = 'document'

    id = Column(BigInteger, primary_key=True)
    document_id = Column(VARCHAR(222))
    title = Column(TEXT)
    publish_date = Column(VARCHAR(255))
    content = Column(TEXT)
    add_time = Column(VARCHAR(255))
    create_time = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), comment='创建时间')
    update_time = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)
    name = Column(VARCHAR(255))
    date = Column(VARCHAR(255))


class DocumentAuthor(Base):
    __tablename__ = 'document_author'

    id = Column(BigInteger, primary_key=True)
    author_id = Column(BigInteger)
    document_id = Column(BigInteger)
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)
    add_time = Column(VARCHAR(255))


class DocumentJournal(Base):
    __tablename__ = 'document_journal'

    id = Column(BigInteger, primary_key=True)
    journal_id = Column(BigInteger)
    document_id = Column(BigInteger)
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)
    add_time = Column(VARCHAR(222))


class Experiment(Base):
    __tablename__ = 'experiment'

    experiment_id = Column(BigInteger, primary_key=True)
    dataset_id = Column(BigInteger)
    status = Column(Integer)
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    model_name = Column(String(45, 'utf8mb4_general_ci'))
    process_config = Column(String(255, 'utf8mb4_general_ci'))
    process_log_path = Column(VARCHAR(255), comment='语言')
    model_config = Column(String(255, 'utf8mb4_general_ci'))
    train_log_path = Column(VARCHAR(255), comment='机构')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')


class Journal(Base):
    __tablename__ = 'journal'

    id = Column(BigInteger, primary_key=True)
    journal_id = Column(BigInteger)
    journal_name = Column(VARCHAR(255))
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)
    add_time = Column(VARCHAR(222))


class TMapT(Base):
    __tablename__ = 't_map_t'

    id = Column(BigInteger, primary_key=True)
    top_real_id = Column(VARCHAR(222))
    parent_id = Column(VARCHAR(255))
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)
    value = Column(VARCHAR(222))
    add_time = Column(VARCHAR(222))


class TMapW(Base):
    __tablename__ = 't_map_w'

    id = Column(BigInteger, primary_key=True)
    word_id = Column(BigInteger)
    word_key = Column(VARCHAR(255))
    word_value = Column(VARCHAR(255))
    topic_id = Column(VARCHAR(255))
    topic_name = Column(VARCHAR(255))
    category_id = Column(Integer)
    add_time = Column(VARCHAR(255))
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)


class Topic(Base):
    __tablename__ = 'topic'

    id = Column(BigInteger, primary_key=True)
    name = Column(VARCHAR(255))
    topic_id = Column(VARCHAR(255))
    value = Column(VARCHAR(255))
    add_time = Column(VARCHAR(255))
    create_time = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), comment='创建时间')
    update_time = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)
    x = Column(VARCHAR(225))
    y = Column(VARCHAR(225))


class WMapW(Base):
    __tablename__ = 'w_map_w'

    id = Column(BigInteger, primary_key=True)
    map_category = Column(Integer)
    word_id = Column(BigInteger)
    map_id = Column(BigInteger)
    add_time = Column(VARCHAR(255))
    value = Column(VARCHAR(222))
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger)


class Word(Base):
    __tablename__ = 'word'

    id = Column(BigInteger, primary_key=True, nullable=False)
    word_id = Column(BigInteger)
    word_key = Column(VARCHAR(255))
    add_time = Column(VARCHAR(255))
    create_time = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), comment='创建时间')
    update_time = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')
    dataset_id = Column(BigInteger, primary_key=True, nullable=False)
