from datamanager import dbhelper, interactions_composer
import os
from constants import DATA_DIR


def demo_database():
    interactions = 'test1.interactions'
    interactions_path = os.path.join(DATA_DIR, interactions)
    
    sql_db_name = 'abc.sqlite'
    db = dbhelper.get_db(sql_db_name)
    
    # Add interactions file to db
    db.add(interactions_path)
    
    # Get data from db
    print db.getall(lastname='"wright"', grammarlevel='"G1"')     
    db.close()
    
def demo_interactions_composer():
    interactions = 'test1.interactions'
    interactions_path = os.path.join(DATA_DIR, interactions)
    
    sql_db_name = 'abc.sqlite'
    db = dbhelper.get_db(sql_db_name)
    
    rows = db.getall(lastname='"wright"', grammarlevel='"G1"')  
    interactionsfile_header = get_interactionfile_header()
    
    create_interactionsfile(interactionsfile_header, rows)
    
    db.close()   

if __name__ == '__main__':

    #demo_database()
    demo_interactions_composer()
