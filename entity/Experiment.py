from sqlalchemy import BigInteger, Column, Date, DateTime, Integer, String, text
from sqlalchemy.dialects.mysql import TEXT, VARCHAR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

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