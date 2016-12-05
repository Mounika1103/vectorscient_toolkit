import pytest

from ..sources.data import LocalFileIpInfoProvider
from ..sources.data import IpInfoError
from ..utils import data_path


IP_DATA_LOCAL_PATH = data_path("ipdata.bin")


@pytest.fixture
def ip_db():
    provider = LocalFileIpInfoProvider(IP_DATA_LOCAL_PATH)
    return provider


def test_getting_ip_info(ip_db):
    members = "IP", "Domain", "City", "Latitude", "Longitude", "Country"
    ip_info = ip_db.get_info("63.152.126.77")
    for member in members:
        msg = "Member '%s' was expected but not found" % member
        assert member in ip_info, msg


def test_ip_info_malformed_argument(ip_db):
    invalid_inputs = 10.00, "1.1.1.", "10.10.10.999"
    for value in invalid_inputs:
        with pytest.raises(IpInfoError):
            ip_db.get_info(value)
