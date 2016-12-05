import argparse

from vectorscient_toolkit.transformation.web import WebDataTransformation


def parse_args():
    parser = argparse.ArgumentParser(description="Web data trasnformation", prog="transform.py")
    parser.add_argument('-db', '--database', type=str, required=True, help="client\'s database name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    web = WebDataTransformation(database=args.database)
    web.process()