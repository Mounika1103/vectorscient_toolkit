import argparse

from vectorscient_toolkit.sources.data2sql import CrmDataSource


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process CRM structured data", prog="main.py")
    parser.add_argument('-db', '--database',
                        type=str, required=True, help="client\'s database name")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        crmdata = CrmDataSource(database=args.database)
        crmdata.update()

    except Exception as e:
        print("Error: {}".format(e))
