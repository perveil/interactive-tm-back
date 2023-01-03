#!/usr/bin/python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pymysql
import schedule
import time
from enum import Enum
import shlex
import subprocess

import os
from entity.Dataset import Dataset
from entity.Experiment import Experiment

from  utils import convertStr2Dict, convertStr2Arg
from process_module.process import process
Expriment_table = "experiment"
Dataset_table = "dataset"
Database_name = "wtd-data"


class FILED(Enum):
    ID = 0
    DSID = 1
    DSNAME = 2
    DSINPATH = 3
    DSOUTPATH = 4
    DSSTATUS = 5

class STATUS(Enum):
    UPLOADED = 0  # 已上传
    PREPROCESSING = 1 # 预处理中
    PREPROCESSFAILED = 2 # 预处理失败
    PREPROCESSSUCCESSED = 3 # 预处理成功
    WAITFORTRAIN = 4 # 前端点击train 的按钮之后
    TRAINING = 5 # 训练中 
    TRAINFAILED = 6 # 训练失败
    TRAINSUCCESSED = 7 # 训练成功

#### mysql db operate

def connect_mysql(host:str, user:str, password:str, database:str):
    return create_engine(
                url= f"mysql://{user}:{password}@{host}:3306/{database}",
                echo=True,
                pool_size=8,
                pool_recycle=60*30
            )


def query_unprocessed_task(engine):
    DbSession = sessionmaker(bind=engine)
    session = DbSession()
    unprocessed_experiments = session.query(Experiment).filter_by(status = 0).all()
    return unprocessed_experiments



## task schedule logic

def process_task(task:Experiment, engine):
    DbSession = sessionmaker(bind=engine)
    session = DbSession()
    bind_dataset:Dataset = session.query(Dataset).filter_by(dataset_id = task.dataset_id).all()
    #### 首先指定实验的输入输出路径
    input_data_path = bind_dataset.input_path
    output_data_path = f"./output/{bind_dataset.name}-experiment-{task.experiment_id}/"
    process_log = f"./log/{bind_dataset.name}-experiment-process-{task.experiment_id}.log"
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    
    process_config_str = task.process_config
    process_config_arg = convertStr2Arg(process_config_str)

    cmd_parts = [
                'python ./process_module/process.py',
                f'--data_name {bind_dataset.name}',
                f"--input_path {input_data_path}",
                f"--input_file data.excel",
                f"--output_path {output_data_path}",
                f"--lang {bind_dataset.language}",
    ] + process_config_arg
    is_fail = 0
    try:
        subprocess.Popen(cmd_parts, stdout=process_log, stderr=process_log,encoding='utf-8')
    except Exception as e:
        print("执行出错")
        is_fail = 1

    if is_fail:
        session.query(Experiment).filter_by(
            experiment_id = task.experiment_id
            ).update({
                'process_log_path': process_log,
                "status":2})
    else:
        session.query(Experiment).filter_by(
            experiment_id = task.experiment_id
            ).update({
                "process_log_path": process_log,
                "status":3
            })

    




    

def schedule_task(engine):
    tasks = query_unprocessed_task(engine)
    if len(tasks) < 1 :
        return
    for task in tasks :
        process_task(task, engine)



## main logic here

engine = connect_mysql(host='localhost',user='root',password='matrix666',database='wtd-data')
robin_gap_time = 2

schedule.every(robin_gap_time).seconds.do(schedule_task, engine)
while True:
    schedule.run_pending()
print("bye")
