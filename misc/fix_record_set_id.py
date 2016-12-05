import argparse

from vectorscient_toolkit.sources.fixes import FixRecordSetId


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix record_set_id", prog="fix_record_set_id.py")
    parser.add_argument('-db', '--database', type=str, required=True)
    parser.add_argument('-dbt', '--dbtype', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    try:
        landing_tbl = FixRecordSetId(
            database=args.database, db_type=args.dbtype)
        landing_tbl.fix_record_set_id()
    except Exception as e:
        print ("Error: {}".format(e))