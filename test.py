from datamanager import dbhelper, interactions_composer, constants, util
import os
DATA_DIR = constants.DATA_DIR

def demo_adding_interactions_to_database():
    interactions = 'test1.interactions'
    interactions = os.path.join(DATA_DIR, interactions)

    sql_db_name = 'output.sqlite'
    db = dbhelper.getdb(sql_db_name)

    # Add interactions file to db
    db.add(interactions)
    db.close()

def demo_fetching_from_database():
    
    # Get data from db
    sql_db_name = 'output.sqlite'
    db = dbhelper.getdb(sql_db_name)
    for row in db.getall(conf_mode='>=', conf=500):
        try:
            print row
        except:
            pass
    db.close()


def demo_interactions_composer():
    
    sql_db_name = 'output.sqlite'
    db = dbhelper.getdb(sql_db_name)
    rows = db.getall(conf_mode='>=', conf=500)
    db.close()
    
    interactionsfile_header = interactions_composer.get_interactionfile_header()
    interactions_composer.create_interactionsfile(interactionsfile_header, rows)


    
    
if __name__ == '__main__':

    #demo_adding_to_database()
    #demo_fetching_from_database()
    #demo_interactions_composer()
    
    # Bulk addition of interactions to db
    sql_db_name = 'output.sqlite'
    db = dbhelper.getdb(sql_db_name)
    with open('all_int.txt', 'r') as f:
        interactions = [line.rstrip() for line in f]
        interactions = filter(lambda interactionfilename: 'TWM' not in interactionfilename, interactions)
        
    for interaction in interactions:
        interaction_filename = os.path.join(DATA_DIR, interaction)
        db.add(interaction_filename)
    db.close()
    

    
