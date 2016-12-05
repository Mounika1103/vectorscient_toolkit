from functools import singledispatch, update_wrapper
from collections import namedtuple, Set, Mapping
from string import ascii_lowercase
from copy import deepcopy
import itertools
import json

from urllib3.exceptions import RequestError, MaxRetryError
from geopy.geocoders import Nominatim
from geopy.exc import GeopyError
from xlrd import XLRDError
import pandas as pd
import numpy as np
import urllib3

from .config import *


__all__ = ["try_to_request_json_data", "try_to_read_local_data",
           "validate_ip", "shallow_flatten", "resolve_lat_long"]


def data_path(path: str):
    base_dir = os.environ["VECTORSCIENT_TOOLKIT_DATA"]
    full_path = os.path.abspath(os.path.join(base_dir, path))
    return full_path


def res(filename):
    """
    Resolves resource name into full path to that resource.

    Args:
        filename: A resource name (with extension).

    Returns:
        str: A full path to the required resource.
    """
    name, ext = os.path.splitext(filename)
    folder = os.path.dirname(os.path.join(__file__))

    if ext in (".png", ".jpeg"):
        return os.path.join(folder, "resources", "images", filename)

    elif ext in (".txt", ".html"):
        return os.path.join(folder, "resources", "texts", filename)

    return None


def merging_update(first, second):
    """
    Updates first dictionary with missing keys from the second one.
    """
    def merge(dct, merge_dct):
        for k, v in merge_dct.items():
            if (k in dct and isinstance(dct[k], dict)
                    and isinstance(merge_dct[k], Mapping)):
                merge(dct[k], merge_dct[k])
            else:
                dct[k] = merge_dct[k]

    if first is None and second is not None:
        return second

    if second is None and first is not None:
        return first

    if (first, second) == (None, None):
        return {}

    acc = deepcopy(first)
    merge(acc, second)
    return acc


def method_dispatch(func):
    """
    Analogue of single dispatch decorator for instance methods.
    """
    dispatcher = singledispatch(func)

    def wrapper(*args, **params):
        return dispatcher.dispatch(args[1].__class__)(*args, **params)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def try_to_request_json_data(url: str, timeout: float=None) -> dict:
    """
    Makes attempt to retrieve JSON data from remote host.
    """
    http = urllib3.PoolManager(timeout=timeout)
    error = None

    try:
        response = http.request('GET', url)
        result = json.loads(response.data.decode('utf8'))
    except (RequestError, MaxRetryError, ValueError) as e:
        error, result = str(e), {}

    return error, result


def try_to_read_local_data(reader):
    """
    Makes attempt to read local file with data using provided reader.
    """
    try:
        data = reader()
        return None, data

    except (FileNotFoundError, XLRDError) as e:
        error = "Cannot read local data: %s" % e
        return error, None


def validate_ip(ip_address) -> bool:
    """
    Validates provided IP address.
    """

    if isinstance(ip_address, str):
        ip = [int(b) for b in ip_address.split('.') if b]
    elif isinstance(ip_address, Set):
        ip = list(ip_address)
    else:
        return False

    if len(ip) != 4:
        return False

    if not all(0 <= byte <= 255 for byte in ip):
        return False

    return True


def shallow_flatten(nested_json: dict):
    """
    Converts JSON structure with compound fields into list of flattened
    dictionaries. Can only handle objects with single nesting level.
    """

    def compound_items(d, tp):
        return [k for k in d.keys() if isinstance(d[k], tp)]

    list_keys = compound_items(nested_json, list)
    dict_keys = compound_items(nested_json, dict)
    new = {k: v for k, v in nested_json.items()
           if k not in list_keys and k not in dict_keys}
    tables = [[new]]

    for key in dict_keys:
        nested_json[key] = [nested_json[key]]
        list_keys.append(key)

    for key in list_keys:
        table = []
        for nested_item in nested_json[key]:
            row = {key + "_" + k: v for k, v in nested_item.items()}
            table.append(row)
        tables.append(table)

    rows = []

    for tup in itertools.product(*[t for t in tables if t]):
        joined = {}
        for item in tup:
            joined.update(item)
        rows.append(joined)

    return rows


GeoPoint = namedtuple("GeoPoint", ["lat", "lon"])


def resolve_lat_long(address: str,
                     geo_locator_factory=Nominatim) -> GeoPoint:
    """
    Returns (latitude, longitude) values for provided address

    Args:
        address: string of characters representing address
        geo_locator_factory: one of available geo locators
    """
    geo_locator = geo_locator_factory()
    location = None

    try:
        location = geo_locator.geocode(address)
    except GeopyError as e:
        pass

    if not location:
        return GeoPoint(lat=np.NaN, lon=np.NaN)

    return GeoPoint(lat=location.latitude, lon=location.longitude)


def inverse_transform(x_norm, x_orig):
    df_norm = pd.DataFrame(x_norm)
    df_orig = pd.DataFrame(x_orig)
    inv_transform = df_norm*(df_orig.max() - df_orig.min()) + df_norm.mean()
    return inv_transform


def normalize(data, low: float = 0, high: float = 1):
    """
    Normalizes data frame values between [low, high] range.

    Args:
        data (object) - an array or dataset to be normalized
        low (float) - lower bound of normalization range
        high (float) - upper bound of normalization range
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        arr = np.array(data)
    else:
        arr = data

    m = arr.min()
    norm = (arr - m) / (arr.max() - m)
    rescaled = norm * (high - low) + low
    return rescaled


def remove_extra_symbols(text: str, sentence=False, allowed=None):
    """
    Removes all non ASCII symbols from the string.
    """
    if allowed is None:
        allowed_symbols = set(ascii_lowercase + '-/')
    else:
        allowed_symbols = allowed

    def remove(word):
        symbols = [letter for letter in word if letter in allowed_symbols]
        ascii_only = "".join(symbols)
        return ascii_only

    if not sentence:
        return remove(text)

    cleaned = [remove(word) for word in text.split()]
    non_emtpy = [w for w in cleaned if w]
    return " ".join(non_emtpy)


def clean_contractions(text: str):
    """
    Replaces possible verbs contractions with their full forms.
    """
    contraction = {
        "don't": "do not",
        "do n't": "do not",
        "dont": "do not",
        "doesn't": "does not",
        "does n't": "does not",
        "doesnt": "does not",
        "didn't": "did not",
        "did n't": "did not",
        "didnt": "did not",
        "can not": "cannot",
        "cant": "cannot",
        "can't": "cannot",
        "ca n't": "cannot",
        "i've": "i have",
        "we've": "we have",
        "you've": "you have",
        "you're": "you are",
        "i'm": "i am",
        "wouldn't": "would not",
        "wouldnt": "would not",
        "couldn't": "could not",
        "couldnt": "could not"
    }
    cleaned = text
    for word, replacement in contraction.items():
        cleaned = cleaned.replace(word, replacement)
    return cleaned


# TODO: replace with native pandas to_sql calls
class DataExporter:

    template = """
INSERT INTO {database_table}
({columns})
VALUES
{data_rows}
"""

    def __init__(self, data: pd.DataFrame, **params):
        self._data = data
        self._table_name = params.get("table_name", None)
        self._cursor = params.get("cursor", None)
        self._connector = params.get("connector", None)
        self._connection_config = params.get("connection_config", {})

    def to_sql(self):
        """
        Converts data frame into SQL-query for MySQL database. Don't to use
        with untrusted input sources.
        """
        data = self._data
        table_name = self._table_name

        columns = ",\n".join(data.columns.tolist())
        data_rows = ",\n".join([str(row) for row in data.itertuples()])

        query = DataExporter.template.format(
            database_table=table_name,
            columns=columns,
            data_rows=data_rows)

        return query

    def export(self, query: str):
        cur = self._cursor

        if cur is not None:
            cur.execute(query)
            cur.commit()

        else:
            with self._connector.connect(**self._connection_config) as cursor:
                cursor.execute(query)
                cursor.commit()


if PATCH_IP_ADDRESSES:
    from ipaddress import IPv4Address
    import random

    USA_IP_RANGES_RAW = (
        ('5.0.0.0', '15.255.255.255'),
        ('22.0.0.0', '22.255.255.255'),
        ('33.0.0.0', '33.255.255.255'),
        ('34.0.0.0', '34.255.255.255'),
        ('44.0.0.0', '44.255.255.255'),
        ('52.0.0.0', '52.31.255.255'),
        ('63.0.0.0', '63.63.255.255'),
        ('65.0.0.0', '65.15.255.255'),
        ('65.128.0.0', '65.159.255.255'),
        ('66.72.0.0', '66.73.255.255'),
        ('70.144.0.0', '70.159.255.255')
    )

    USA_IP_RANGES = [(IPv4Address(start_ip), IPv4Address(end_ip))
                     for start_ip, end_ip in USA_IP_RANGES_RAW]

    def patch_ip(ip_address: str):
        """
        IP address resolution issue hot fix. Ensures that octets are within
        appropriate range of values.
        """
        ip = IPv4Address(ip_address)

        for start_ip, end_ip in USA_IP_RANGES:
            if not (start_ip <= ip <= end_ip):
                continue

            _, _, third, fourth = [int(x) for x in ip_address.split('.')]
            if third == 0 and fourth == 0:
                ip += random.randint(257, 256**2 - 1)

            return str(ip)

        # if not in range - create from scratch
        start_ip, end_ip = random.choice(USA_IP_RANGES)

        while True:
            new_ip = start_ip + random.randint(1, 10 * 255**2)
            if new_ip <= end_ip:
                break

        return str(new_ip)

else:

    def patch_ip(ip_address: str):
        return ip_address  # don't do anything in release mode
