Initial started writing my own sqlite interface called db_helper
It turned out that I had to write too many things ... say define >=, <=,
etc behaviour. Turned out to be too much work.

A co-worker introduced me to peewee an orm framework in python

Use orm_helper for data fetching and to add interactions using db_helper and use 
the add method in it.

Note: The requirements.txt has too many modules because I didb't use virtualenv. All of them are not really needed. Just peewee is.
