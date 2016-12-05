import argparse
from vectorscient_toolkit.config import LOCATION_DB_PATH

from vectorscient_toolkit.sources.data import WebAnalyticsSource
from vectorscient_toolkit.sources.data2sql import WebDataSource
from vectorscient_toolkit.sources.mixins import VSWebRequest


def parse_args():
    parser = argparse.ArgumentParser(description="Save Web data", prog="main.py")
    parser.add_argument('-db', '--database', type=str, required=True, help="client\'s database name")
    parser.add_argument('-s', '--subdomain', type=str, required=True, help="client\'s subdomain")

    # ENABLE/DISABLE PRE-ALGORITHM PROCESS
    parser.add_argument('--process', dest='process_rl', action='store_true',
        help="Enable the Pre-Algorithm process")
    parser.add_argument('--no-process', dest='process_rl', action='store_false',
        help="Disable the Pre-Algorithm process")
    parser.set_defaults(process_rl=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    request = VSWebRequest(db=args.database, subdomain=args.subdomain)
    url = request.build_uri()
    dbpath = LOCATION_DB_PATH

    try:
        websource = WebAnalyticsSource(url, database_path=dbpath)
        websource.prepare()
        if websource.ready:
            web = WebDataSource(database=args.database)
            web.save_to_db(websource.data)
        else:
            print ("error: cannot prepare source")

    except Exception as e:
        print ("error: %s" % e)