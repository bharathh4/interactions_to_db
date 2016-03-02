import sqlite3
from util import process, get_rows

def compose_single_sqlite_condition(params, k, v):
    sql_condition = []
    if type(v) in [int, float]:
        sign = params[k + '_' + 'mode'] # Generate conf_mode and get the value of sign by composing param[conf_mode]
        sql_condition = '%s%s%s' % (k, sign, str(v))
    else:
        sql_condition = '%s="%s"' % (k, v)
        
    return sql_condition
    
def compose_all_sqlite_conditions(params):
    conditions = []
    for k, v in sorted(params.iteritems()):
        keys_to_avoid = ['self']
        values_to_avoid = [None, '>=', '<=', '>', '<', '='] # designed to skip conf_mode='>='. Anything that has signs in them
        if k not in keys_to_avoid and v not in values_to_avoid:
            sql_condition = compose_single_sqlite_condition(params, k, v)
            conditions.append(sql_condition)            
    return conditions

def compose_sqlite_query(params, tablename):

    sql_header_statement = '''SELECT * FROM %s WHERE''' % (tablename)
    conditions = compose_all_sqlite_conditions(params)             
    sql_statement = sql_header_statement + ' ' + " and ".join(conditions)
    print sql_statement
    return sql_statement


def getdb(sql_db_name):
    db = DB(sql_db_name)
    return db


class DB():

    def __init__(self, sql_db_name):
        self.sql_db_name = sql_db_name
        self.conn = sqlite3.connect(sql_db_name)
        # Check if this is right
        self.conn.text_factory = str


    def close(self):
        print 'Comitting and closing the database'
        self.conn.commit()
        self.conn.close()

    def add(self, interactions):
        print 'Adding interactions to database'
        tablename = 'transcriptions'
        self._create(tablename)
        for row in get_rows(interactions):
            try:
                self._addrow(row, tablename)
            except:
                print 'Could not add row. Is the callsre naming format correct ?'
                        
    def getall(self, transcript_si=None, transcript=None, decode_si=None,
               decode=None, conf_mode=None, conf=None, decode_time_mode=None, decode_time=None, callsrepath=None, acoustic_model=None,
               date=None, time=None, milliseconds=None, grammarlevel=None, firstname=None, lastname=None,
               oration_id=None, chain=None, store=None, exactinteractionfilerow=None):

        # Get function parameters
        params = locals().copy()
        tablename = 'transcriptions'
        sqlite_query = compose_sqlite_query(params, tablename)
        print 'Getting all db entries for parameters'
        c = self.conn.cursor()
        rows = [row[-1] for row in c.execute(sqlite_query)]    
        #rows = [row for row in c.execute(sqlite_query)]    
        if rows:
            return rows
        else:
            print 'No such data in db for the given parameters'
       
        
    def _get_table_names(self):
        c = self.conn.cursor()
        rows = c.execute(
            '''SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;''')
        return [row for row in rows.fetchall()]
        
    def _create(self, tablename):
        # Creating table if it does not exist in db
        c = self.conn.cursor()
        if not self._check_if_table_exists(tablename):
            sql_statement = '''CREATE TABLE %s (transcript_si, transcript, decode_si,
             decode, conf, decode_time, callsrepath, acoustic_model, 
             date, time, milliseconds, grammarlevel, firstname, lastname, 
             oration_id, chain, store, exactinteractionfilerow)''' % (
                tablename)
            c.execute(sql_statement)
            
    def _addrow(self, row, tablename):
        c = self.conn.cursor()

        
        (transcript_si, transcript, decode_si,
             decode, conf, decode_time, callsrepath, acoustic_model,
             date, time, milliseconds, grammarlevel, firstname, lastname,
             oration_id, chain, store) = process(row)

        # There might a better way. Breaking out of a function if the transcript column is empty.
        # This files without transcription are skipped
        if transcript in ['', '""', ' ', None]:
            print 'No transcription for %s'
            print callsrepath
            return False
            
        # Insert row into database
        # Check if row exists. If yes don't add. If not add
        datarow_exist_status = self._check_if_row_exists(
                time, milliseconds, firstname, lastname, grammarlevel, tablename)
        if datarow_exist_status:
            print 'The data row you are trying to add might already exist'
        else:
            print 'Adding data row to database'
            sql_statement = '''INSERT INTO %s VALUES ("%s", "%s", "%s", "%s", %d, %d, "%s", "%s", %s, %s, %s, "%s", "%s", "%s", "%s", "%s", "%s", "%s")''' % (
                    tablename, transcript_si, transcript, decode_si,
                    decode, int(conf), int(
                        decode_time), callsrepath, acoustic_model,
                    date, time, milliseconds, grammarlevel, firstname, lastname,
                    oration_id, chain, store, row)
            try:
                c.execute(sql_statement)
            except:
                print 'Could not enter row to database'

    def _check_if_table_exists(self, tbname=None):
        tablenames = self._get_table_names()

        if tablenames:
            if tbname in tablenames[0]:
                print 'Table exists in database'
                return True
            else:
                print 'Table does not exist in database'
                return False
        else:
            print 'No Tables found in the database'

    def _check_if_row_exists(self, time, milliseconds, firstname, lastname, grammarlevel, tablename):
        # Using time, firstname, lastname and grammarlevel as to check
        c = self.conn.cursor()
        sql_statement = "SELECT time, milliseconds, firstname, lastname, grammarlevel  FROM %s" % (
            tablename)
        for row in c.execute(sql_statement):
            if (int(time), int(milliseconds), firstname, lastname, grammarlevel) == (row):
                return True
        else:
            return False


