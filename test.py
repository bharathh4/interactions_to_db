from datamanager import dbhelper, interactions_composer, constants, util, orm_helper, analyse
DATA_DIR = constants.DATA_DIR
import os


def demo_adding_interactions_to_database():
    interactions = 'test1.interactions'
    interactions = os.path.join(DATA_DIR, interactions)

    sql_db_name = 'output23.sqlite'
    db = dbhelper.getdb(sql_db_name)

    # Add interactions file to db
    db.add(interactions)
    db.close()

# Using home grown 
def demo_fetching_from_database():
    
    # Get data from db
    #sql_db_name = 'output.sqlite'
    sql_db_name = 'output23.sqlite'
    db = dbhelper.getdb(sql_db_name)
    
    rows= []
    for row in db.getall(conf=1000, conf_mode='<='):
        if 'CLIP' in row.split(',')[1]:
            rows.append(row)
    print (rows)
            
    db.close()

def demo_fetching_from_database_using_orm():
    sql_db_name = 'output_peewee_integration.sqlite'
    Transcriptions = orm_helper.Transcriptions
    for row in Transcriptions.select().where(Transcriptions.conf > 998):
        print row.store, row.conf
   
def demo_interactions_composer():
    
    sql_db_name = 'output.sqlite'
    db = dbhelper.getdb(sql_db_name)
    rows = db.getall(conf_mode='=', conf=999)
    print type(rows)
    db.close()
    
    interactionsfile_header = interactions_composer.get_interactionfile_header()
    interactions_composer.create_interactionsfile(interactionsfile_header, rows)

def demo_interactions_composer_using_orm():
    db_path = 'C:\Users\TheatroIT\Documents\Scripts2\interactions_to_db\output_peewee_integration.sqlite'
    Transcriptions = orm_helper.Transcriptions
    rows = [row.exactinteractionfilerow for row in Transcriptions.select().where(Transcriptions.conf > 998)]
    
    interactionsfile_header = interactions_composer.get_interactionfile_header()
    interactions_composer.create_interactionsfile(interactionsfile_header, rows)
    
    
def bulk_add_interactions():
    # Bulk addition of interactions to db
    
    sql_db_name = 'output_peewee_integration.sqlite'
    db = dbhelper.getdb(sql_db_name)
    
    # get interaction file names from all_int.txt
    with open('dataall_int.txt', 'r') as f:
        interactions = [line.rstrip() for line in f]
        interactions = filter(lambda interactionfilename: 'TWM' not in interactionfilename, interactions)
        
    for interaction in interactions:
        interaction_filename = os.path.join(DATA_DIR, interaction)
        db.add(interaction_filename)
    db.close()
       
    
if __name__ == '__main__':

    #demo_adding_interactions_to_database()
    #demo_fetching_from_database()
    #demo_interactions_composer()
    #demo_fetching_from_database_using_orm()
    #demo_interactions_composer_using_orm()
    
    
    
    '''
    db_path = 'C:\Users\TheatroIT\Documents\Scripts2\interactions_to_db\output_peewee_integration.sqlite'
    Transcriptions = orm_helper.Transcriptions
    #rows = [row.exactinteractionfilerow for row in Transcriptions.select().where(Transcriptions.chain == 'TheContainerStore')]
    #rows = [row.exactinteractionfilerow for row in Transcriptions.select()]
    #rows = [row.exactinteractionfilerow for row in Transcriptions.select().where(Transcriptions.chain == 'BassPro')]
    #rows = [row.exactinteractionfilerow for row in Transcriptions.select().where(Transcriptions.chain == 'Cabelas')]

    interactionsfile_header = interactions_composer.get_interactionfile_header()
    interactions_composer.create_interactionsfile(interactionsfile_header, rows)
    '''
    
    
    #filename = 'data/HDC-7135_2016_03_30_31_04_01_ALL_v2_troubled2.Interactions'
    #filename = 'data/HDC-7135_20160422-26_MANAGERS.Interactions'
    #filename = 'data/CAB-ALN_20151020_ALL.Interactions'
    #filename = 'data/TCS-AUS_20150905_ALL.Interactions'
    #filename = 'data/BPS-OW39_20150908_ALL.Interactions'
    #analyse.main(filename, name='veloso')
    #filename = 'data/just_carlos30_1.Interactions'
    
    
    
    #filename = 'data/HDC-7135_20160430-0501_ALL.Interactions'
    #filename = 'data/HDC-7135_20160430-0501_ALL_grammar-mods.Interactions'
    #analyse.main(filename)
    
    
    '''
    troubled_users = ["carlos veloso"]
    #filename = 'data/HDC-7135_20160416_ALL.Interactions'
    #filename = 'data/HDC-7135_2016_03_30_31_04_01_ALL_v2.Interactions'
    filename = 'data/HDC-7135_20160430-0501_ALL - JUST CARLOS.Interactions'
    interactions_composer.create_interactionsfile_for_users(troubled_users, filename)
    '''
    
    '''
    filename = 'data/HDC-7135_20160430-0501_ALL.Interactions'
    interactions_composer.create_interactionsfile_for(filename, needed_items=['micheal', 'mike', 'michael'], attribute='transcript')
    '''
    
    filenames = ['data/HDC-7135_20160430-0501_ALL.interactions', 'data/HDC-7135_20160430-0501_ALL_grammar-mods.interactions']
    interactions_composer.create_interactionsfile_for_movement_change(filenames)
