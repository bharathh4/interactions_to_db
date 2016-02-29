import dbhelper
import os
from constants import DATA_DIR, OUTPUT_INTERACTIONFILENAME


def get_interactionfile_header():

    return '''|0|0|0|## Version 4.0.0 ##
        <settings>
            <grammarlist>
            </grammarlist>
            <filterlist>
                <filtergroup name="New Group">
                </filtergroup>
            </filterlist>
        </settings>'''

def create_interactionsfile(interactionsfile_header, rows):
    # Creating an interaction file
    with open(OUTPUT_INTERACTIONFILENAME, 'w') as f:
        f.write(interactionsfile_header+'\n')
        for row in rows:
            f.write(row)
    

def test():
    interactions = 'test1.interactions'
    interactions_path = os.path.join(DATA_DIR, interactions)
    
    sql_db_name = 'abc.sqlite'
    db = dbhelper.get_db(sql_db_name)
    
    rows = db.getall(lastname='"wright"', grammarlevel='"G1"')  
    interactionsfile_header = get_interactionfile_header()
    # 
    create_interactionsfile(interactionsfile_header, rows)
    
    db.close()   
 
if __name__ == '__main__':
    #test_query_compose_interactions_file()
    pass
    