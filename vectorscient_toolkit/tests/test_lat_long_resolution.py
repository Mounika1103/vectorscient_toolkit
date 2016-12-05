import numpy as np

from ..utils import resolve_lat_long, GeoPoint


def test_full_address_resolution():
    loc = resolve_lat_long("525 S. Lexington Ave")
    assert isinstance(loc, GeoPoint)
    assert loc.lat != 0 and loc.lon != 0


def test_non_existent_address():
    loc = resolve_lat_long("There is no such place")
    assert loc.lat is np.NaN
    assert loc.lon is np.NaN
