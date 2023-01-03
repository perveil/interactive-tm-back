#!/usr/bin/python

""" sql to create 
create table dataset_task
(
    id                  int auto_increment,
    dataset_id          int                     not null,
    dataset_name        varchar(512) default '' not null,
    dataset_input_path  varchar(512)            null,
    dataset_output_path varchar(512)            null,
    dataset_status      tinyint      default 0  not null,
    constraint dataset_task_pk
        primary key (id)
);

create unique index dataset_task_dataset_id_uindex
    on dataset_task (dataset_id);

"""

import pymysql
import schedule
import time
from enum import Enum


Expriment_table = "experiment"
Dataset_table = "dataset"


class FILED(Enum):
    ID = 0
    DSID = 1
    DSNAME = 2
    DSINPATH = 3
    DSOUTPATH = 4
    DSSTATUS = 5

class STATUS(Enum):
    CREATED = 0  # 已上传
    PREPROCESSING = 1 # 预处理中
    PREPROCESSFAILED = 2 # 预处理失败
    PREPROCESSSUCCESSED = 3 # 预处理成功
    WAITFORTRAIN = 4 # 前端点击train 的按钮之后
    TRAINING = 5 # 训练中 
    TRAINFAILED = 6 # 训练失败
    TRAINSUCCESSED = 7 # 训练成功

#### mysql db operate

def connect_mysql(host:str, user:str, password:str, database:str):
    return pymysql.connect(host=host,
                           port=3306,
                           user=user,
                           password=password,
                           database=database,
                           autocommit=True)


def query_task(querry_status: STATUS, con:pymysql.Connect):
    sql_str = "SELECT * FROM  `wtd-data`.dataset"
    # sql_str = ("SELECT * FROM dataset where dataset_status = " + str(STATUS.CREATED.value) + " or dataset_status = " + str(STATUS.WAITFORTRAIN.value))
    print(str(int(time.time())) +" [MYSQL] execute sql : " + sql_str)
    cur = con.cursor()
    cur.execute(sql_str)
    rows = cur.fetchall()
    cur.close()
    return rows

def push_status_process(task:list) :
    status = task[FILED.DSSTATUS.value]
    status += 1
    sql_str = ("update dataset_task set dataset_status = " + str(status) + " where id = " + str(task[FILED.ID.value]))
    print("[MYSQL] execute sql : " + sql_str)
    cur = con.cursor()
    cur.execute(sql_str)
    cur.close()
    task[FILED.DSSTATUS.value] = status
    return task

def update_output(task:list, output_path:str) :
    status = task[FILED.DSSTATUS.value]
    status += 1
    sql_str = ("update dataset_task set dataset_status = " + str(status) + ", dataset_output_path = '" + output_path + "' where id = " + str(task[FILED.ID.value]))
    print("[MYSQL] execute sql : " + sql_str)
    cur = con.cursor()
    cur.execute(sql_str)
    cur.close()
    task[FILED.DSSTATUS.value] = status
    return task


## task schedule logic

def process_task(task:list, con:pymysql.Connect):
    task = push_status_process(task)
    if (STATUS.CREATED.value + 1 == task[FILED.DSSTATUS.value]):
        do_pre_process(task)
    if(STATUS.WAITFORTRAIN.value + 1 == task[FILED.DSSTATUS.value]):
        do_train(task)

def schedule_task(con:pymysql.Connect):
    tasks = query_task(STATUS.CREATED, con)
    if len(tasks) < 1 :
        return
    for task in tasks :
        task = list(task)
        process_task(task, con)


def do_pre_process(task:list):
    print(str(time.time()) + " do pre porcess : " + str(task))
    dataset_name = task[FILED.DSNAME.value]
    input_path = task[FILED.DSINPATH.value]
    # TODO
    # add code here to do pre process
    # 
    # TODO
    # need to update output path using pre process result
    # output_path = ?
    update_output(task,"./")


def do_train(task:list):
    print(str(time.time()) + " do train : " + str(task))
    dataset_name = task[FILED.DSNAME.value]
    input_path = task[FILED.DSINPATH.value]
    output_path = task[FILED.DSOUTPATH.value]
    # TODO
    # add code here to do train
    # 
    # 
    push_status_process(task)



## main logic here

con = connect_mysql(host='localhost',user='root',password='matrix666',database='wtd-data')
robin_gap_time = 2

schedule.every(robin_gap_time).seconds.do(schedule_task, con)
while True:
    schedule.run_pending()
print("bye")
