import dbhelper
import os
from constants import DATA_DIR, OUTPUT_INTERACTIONFILENAME

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
            row = row.encode('ascii', 'ignore').decode('ascii')
            try:
                f.write(row)
            except:
                print 'Could not write this row to interactions file'


def test():
  
    sql_db_name = 'abc.sqlite'
    db = dbhelper.get_db(sql_db_name)

    rows = db.getall(lastname='wright', grammarlevel='G1')
    interactionsfile_header = get_interactionfile_header()
    
    create_interactionsfile(interactionsfile_header, rows)
    db.close()

if __name__ == '__main__':
    test()
    
