import argparse

from vectorscient_toolkit.qfi.PredictAlly_quality_index_calc import QFI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Web data trasnformation", prog="transform.py")
    parser.add_argument('-db', '--database',
                        type=str, required=True, help="client\'s database name")
    parser.add_argument('-pd', '--predrundate',
                        type=str, required=False, help="Prediction Run date")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    qfi = QFI(database=args.database)
    qfi.process(pred_run_date=args.predrundate)
