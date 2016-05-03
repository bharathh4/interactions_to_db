import dbhelper
import os
from constants import DATA_DIR, OUTPUT_INTERACTIONFILENAME

import analyse
analyse.DATA_SOURCE = 'csv'
from analyse import process, get_reader

'''
This file joins rows extracted from a db and forms an interactions file.
'''

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
        f.write(interactionsfile_header + '\n')
        for row in rows:
            
            f.write(row)
            '''
            try:
                #row = row.encode('ascii', 'ignore').decode('ascii')
                f.write(row)
            except:
                print 'Could not write this row to interactions file'
            '''
       
    print 'Done !'

def test():
  
    sql_db_name = 'abc.sqlite'
    db = dbhelper.get_db(sql_db_name)

    rows = db.getall(lastname='wright', grammarlevel='G1')
    interactionsfile_header = get_interactionfile_header()
    
    create_interactionsfile(interactionsfile_header, rows)
    db.close()

    
def create_interactionsfile_for_users(userslist, filename):

    needed_rows = []
    for row in get_reader(filename):
        
        (transcript_si, transcript, decode_si, decode, conf, decode_time,
             callsrepath, acoustic_model,
             date, time, milliseconds, grammarlevel, firstname, lastname, oration_id,
             chain, store) = process(row)
        fullname = '%s %s' %(firstname, lastname)
        if fullname in userslist:
            needed_rows.append(','.join(row)+'\n')
            
    interactionsfile_header = get_interactionfile_header()
    create_interactionsfile(interactionsfile_header, needed_rows)
    
if __name__ == '__main__':
    test()
    
