import dbhelper
import orm_helper
import os
from constants import DATA_DIR, DB_PATH

def get_conf_avg(storename='AUS'):
    Transcriptions = orm_helper.Transcriptions
    confs = [row.conf for row in Transcriptions.select().where(Transcriptions.storename == 'AUS')]
    return sum(confs)
    
def main():
    get_conf_avg(storename='AUS')
    
if __name__ == '__main__':
    main()
    
    