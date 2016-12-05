import argparse

import functools
import os

from pandas.io import sql
import pandas as pd
import MySQLdb


def compute_normalize(x, min_val, max_val, a=1, b=100):
    """
    Formula:
            (b-a)(x - min)
    f(x) = --------------  + a
              max - min

    Sample: Calculate the Index value of this number "5.579767005" by applying the rule.
    [a, b] = [1, 100]
    Min = 3 and max = 8 ; x=5.579
    F(x) = ((100-1)(5.579 – 3) /(8-3) ) + 1 = ((99*2.579)/5) + 1 = 52.0642

    Taken from PredictAlly_quality_index_calc.py (lines 32-56)
    """
    ab_result = b - a
    xmin_result = round(x - min_val, 3)
    maxmin_result = round(max_val - min_val, 3)
    ab_xmin = round(ab_result * xmin_result, 3)
    try:
        result = round(ab_xmin / maxmin_result, 4) + 1
    except ZeroDivisionError:
        return 1
    return result


def request_qfi_table(cur) -> pd.DataFrame:
    cur.execute("SELECT * FROM CRM_TGT_Stage_Fact")
    fields = [x[0] for x in cur.description]
    result = [dict(zip(fields, row)) for row in cur.fetchall()]
    table_data = pd.DataFrame.from_dict(result, orient='columns')
    return table_data


def calculate_statistics(data):

    def fill_stats(aggregated, row):
        key = row.product_code, row.stage_name
        x = row.stage_duration_in_days
        median, minimal, maximal = aggregated.ix[key].values
        qfi_value = compute_normalize(x, minimal, maximal)
        return pd.Series({
            'MIN': minimal,
            'MAX': maximal,
            'MEDIAN': median,
            'S_QFI': qfi_value
        })

    stat_cols = ['MIN', 'MAX', 'MEDIAN', 'S_QFI']
    relevant_only = pd.DataFrame(data[data.exclude_from_stats == 'NO'])
    relevant_only.drop(stat_cols, inplace=True, axis=1)

    main_stats = relevant_only.groupby(['product_code', 'stage_name']).aggregate({
        'stage_duration_in_days': {
            'MIN': min,
            'MAX': max,
            'MEDIAN': 'median'
        }
    })

    df = pd.DataFrame(relevant_only)
    calculate_qfi = functools.partial(fill_stats, main_stats)
    all_stats = df.apply(calculate_qfi, axis=1)

    result = pd.concat([df, all_stats], axis=1)

    reordered = result[[
        'Stage_fact_pk',
        'Opportunity_id',
        'product_code',
        'stage_name',
        'stage_start_date',
        'stage_end_date',
        'stage_duration_in_days',
        'default_probability',
        'exclude_from_stats',
        'average',
        'MIN',
        'MAX',
        'MEDIAN',
        'S_QFI'
    ]]

    return reordered


def update_qfi_table(cursor, data: pd.DataFrame):
    """Get the Stage_fact_pk then update the fields
    """
    data = data.fillna(value=0)

    for index, row in data.iterrows():
        S_QFI = row['S_QFI']
        if S_QFI == 0:
            S_QFI = 1

        sql = """UPDATE CRM_TGT_Stage_Fact
                SET
                    S_QFI={S_QFI},
                    MIN={MIN},
                    MAX={MAX},
                    MEDIAN={MEDIAN}
                WHERE
                    Stage_fact_pk={Stage_fact_pk}
        """.format(Stage_fact_pk=row['Stage_fact_pk'],
                   S_QFI=S_QFI,
                   # S_QFI=row['S_QFI'],
                   MIN=row['MIN'],
                   MAX=row['MAX'],
                   MEDIAN=row['MEDIAN']
                   )
        cursor.execute(sql)


def parse_args():
    parser = argparse.ArgumentParser(description="Sales Stage QFI", prog="sales_stage_qfi.py")
    parser.add_argument('-db', '--database', type=str, required=True, help="client\'s database name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    database = args.database

    connection_config = dict(
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', 'root'),
        passwd=os.environ.get('DB_PASSWORD', 'password'),
        db=database
    )

    con = MySQLdb.connect(**connection_config)
    cursor = con.cursor()
    df = request_qfi_table(cursor)
    stats = calculate_statistics(df)
    update_qfi_table(cursor, stats)
    con.commit()
    cursor.close()
