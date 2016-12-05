# FIXME (Ilia): replace with relative import, now script fails on my machine due
# FIXME (Ilia): to absence of 'settings' module (is it owned by Django app?)
from settings import DATABASE

DB_ENGINE = {
     'mysql': 'mysql+pymysql://{username}:{password}@{host}/',}

#from ...config import SQL_ENGINE


def get_engine(db):
    """ Get the database engine
    """
    conf = DATABASE
    engine = DB_ENGINE[conf['ENGINE']].format(
       username=conf['USERNAME'], password=conf['PASSWORD'], host=conf['HOST'])

    #engine = SQL_ENGINE
    
    return "{engine}{db_name}".format(engine=engine, db_name=db)