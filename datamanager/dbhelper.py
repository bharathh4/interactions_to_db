import sqlite3
from util import process, get_rows

def compose_sqlite_query(params, tablename):

    sql_header_statement = '''SELECT * FROM %s WHERE''' % (tablename)
    
    conditions = []
    for k, v in sorted(params.iteritems()):
        if k is not 'self' and v is not None:
            print k, v
            sql_statement = '%s=%s' % (k, v)
            conditions.append(sql_statement)
            
    sql_statement = sql_header_statement + ' ' + " and ".join(conditions)
    print sql_statement
    
    return sql_statement


def get_db(sql_db_name):
    db = DB(sql_db_name)
    return db


class DB():

    def __init__(self, sql_db_name):
        self.sql_db_name = sql_db_name
        self.conn = sqlite3.connect(sql_db_name)

    def close(self):
        print 'Comitting and closing the database'
        self.conn.commit()
        self.conn.close()

    def get_table_names(self):
        c = self.conn.cursor()
        rows = c.execute(
            '''SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;''')
        return [row for row in rows.fetchall()]

    def check_if_table_exists(self, tbname=None):
        tablenames = self.get_table_names()

        if tablenames:
            if tbname in tablenames[0]:
                print 'Table exists in database'
                return True
            else:
                print 'Table does not exist in database'
                return False
        else:
            print 'No Tables found in the database'

    def check_if_row_exists(self, time, milliseconds, firstname, lastname, grammarlevel, tablename):
        # Using time, firstname, lastname and grammarlevel as to check
        c = self.conn.cursor()
        sql_statement = "SELECT time, milliseconds, firstname, lastname, grammarlevel  FROM %s" % (tablename)
        for row in c.execute(sql_statement):
            if (int(time), int(milliseconds), firstname, lastname, grammarlevel) == (row):
                return True
        else:
            return False

    def add(self, interactions):

        print 'Adding interactions to database'

        # Creating table if it does not exist in db
        c = self.conn.cursor()
        tablename = 'transcriptions'
        if not self.check_if_table_exists(tablename):
            sql_statement = '''CREATE TABLE %s (transcript_si, transcript, decode_si,
             decode, conf, decode_time, callsrepath, acoustic_model, 
             date, time, milliseconds, grammarlevel, firstname, lastname, 
             oration_id, chain, store, exactinteractionfilerow)''' % (
                tablename)
            c.execute(sql_statement)

        for row in get_rows(interactions):

            data = process(row)

            (transcript_si, transcript, decode_si,
             decode, conf, decode_time, callsrepath, acoustic_model,
             date, time, milliseconds, grammarlevel, firstname, lastname,
             oration_id, chain, store) = data

            # Insert row into database
            # Check if row exists. If yes don't add. If not add
            datarow_exist_status = self.check_if_row_exists(time, milliseconds, firstname, lastname, grammarlevel, tablename)
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
                c.execute(sql_statement)

    def getall(self, transcript_si=None, transcript=None, decode_si=None,
             decode=None, conf=None, decode_time=None, callsrepath=None, acoustic_model=None, 
             date=None, time=None, milliseconds=None, grammarlevel=None, firstname=None, lastname=None, 
             oration_id=None, chain=None, store=None, exactinteractionfilerow=None):
             
        # Get function parameters
        params = locals().copy()
        tablename = 'transcriptions'
        sqlite_query = compose_sqlite_query(params, tablename)
        print 'Getting all db entries'
        c = self.conn.cursor()
        return [row[-1] for row in c.execute(sqlite_query)]
            
