from sqlalchemy import BigInteger, Column, Date, DateTime, Integer, String, text
from sqlalchemy.dialects.mysql import TEXT, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
import json

Base = declarative_base()

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
    is_seed_word = Column(Integer, server_default=text("'0'"), comment='是否上传seedword')
    is_word_embedding = Column(Integer, server_default=text("'0'"))
    is_document_label = Column(Integer, server_default=text("'0'"))
    is_document_link = Column(Integer, server_default=text("'0'"))
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(Date, comment='最后修改时间')
    is_delete = Column(Integer, server_default=text("'0'"), comment='逻辑删除')

