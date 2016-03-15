from peewee import *
from constants import DB_PATH
# Modified peewee.py beahviour by setting the conection object's text_factory
database = SqliteDatabase(DB_PATH, **{})

class UnknownField(object):
    pass

class BaseModel(Model):
    class Meta:
        database = database

class Transcriptions(BaseModel):
    acoustic_model = TextField(null=True)
    callsrepath = TextField(null=True)
    chain = TextField(null=True)
    conf = IntegerField(null=True)
    date = TextField(null=True)
    decode = TextField(null=True)
    decode_si = TextField(null=True)
    decode_time = IntegerField(null=True)
    exactinteractionfilerow = BlobField(null=True)
    firstname = TextField(null=True)
    grammarlevel = TextField(null=True)
    id = IntegerField(null=True)
    lastname = TextField(null=True)
    milliseconds = IntegerField(null=True)
    oration = IntegerField(db_column='oration_id', null=True)
    store = TextField(null=True)
    time = TextField(null=True)
    transcript = TextField(null=True)
    transcript_si = TextField(null=True)

    class Meta:
        db_table = 'transcriptions'