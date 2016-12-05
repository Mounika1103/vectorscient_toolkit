"""
The glue module to wire up data input/output and clustering pipeline.
"""
from ..clustering.unsupervised import ClusteringResult
from ...sources.data import DBConnection


def drop_clustering_results_for_run_date(conn: DBConnection, date: str):
    """
    Clears results of clustering for specified pipeline run date.

    Args:
        conn (DBConnection): An instance of database connection object.
        date (str): The date of previous run that should be dropped.
    """
    raise NotImplementedError()


# TODO: add errors checking
def export_to_database(result: ClusteringResult, conn: DBConnection):
    """
    Saves clustering results into database tables.
    """
    steps = {
        "clustered_file_new_pros": {
            "key": "id",
            "op": "update",
            "data": result.clusters
        },
        "clustered_file_new_pros_norm": {
            "key": "id",
            "op": "insert",
            "data": result.data
        },
        "centroids_new_pros": {
            "key": "record_id",
            "op": "insert",
            "data": result.centroids
        }
    }

    for table_name, params in steps.items():
        print("[.] {} table {}...".format(table_name, params["op"]))
        rows = sorted(params["data"].to_dict("records"),
                      key=lambda r: r[params["key"]])
        operation = getattr(conn, params["op"])
        operation(table_name, rows)
        print("[+] Operation completed")


def export_to_csv(result: ClusteringResult, file_name: str, mode='clusters'):
    """
    Saves clustering results into CSV file.
    """
    if mode == 'clusters':
        result.clusters.to_csv(file_name)
    else:
        result.data.to_csv(file_name)
