import dbhelper
import orm_helper
import os, math
from constants import DATA_DIR, DB_PATH

def get_conf_avg(storename='AUS'):
    Transcriptions = orm_helper.Transcriptions
    confs = [row.conf for row in Transcriptions.select().where(Transcriptions.store == storename)]
    return sum(confs)/float(len(confs))
    
def get_conf_variance(storename='AUS'):
    Transcriptions = orm_helper.Transcriptions
    mean = get_conf_avg(storename=storename)
    diff_sqs = [(row.conf - mean) * (row.conf - mean) for row in Transcriptions.select().where(Transcriptions.store == storename)]
    return sum(diff_sqs)/float(len(diff_sqs))
    
def get_sample_row():
    Transcriptions = orm_helper.Transcriptions
    rows = [row for row in Transcriptions.select()]
    return rows[0]
    
def main():
    print math.sqrt(get_conf_variance(storename='AUS'))
    
if __name__ == '__main__':
    main()
    
    