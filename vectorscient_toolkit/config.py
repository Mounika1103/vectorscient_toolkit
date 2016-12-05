import sys
import os


def get_log_folder():
    path = []
    p = sys.platform
    if p.startswith('linux') or p.startswith('darwin'):
        path = ['/', 'var', 'log']
    elif p.startswith('win32'):
        # path = ['C', 'Temp', 'log']
        pass
    else:
        raise SystemError()
    return os.path.abspath(os.path.join(path))


class LoggingParameters:
    STANDARD_LOG_FOLDER = get_log_folder()
    ERROR_OUTPUT_FILE = os.path.join(
        STANDARD_LOG_FOLDER, "engine_error.log")
    REGULAR_OUTPUT_FILE = os.path.join(
        STANDARD_LOG_FOLDER, "engine_regular.log")


########################
# DATABASE CREDENTIALS #
########################

DATABASE = {
    'USERNAME': 'vectorscient',
    'PASSWORD': 'vectorscient456',
    'HOST': '104.197.75.70',
    'ENGINE': 'mysql',
}

##########
# DOMAIN #
##########

DOMAIN = 'vantena.com'


#####################
# LOCATION DATABASE #
#####################
# http://www.ip2location.com/developers/python

LOCATION_DB_PATH = 'data/locationdb/IP-COUNTRY-REGION-CITY-LATITUDE-LONGITUDE-ISP-DOMAIN.BIN'


#######################
# FEBRL CONFIGURATION #
#######################

FEBRL_CONFIG = {
    'python': '/home/vantena/.virtualenvs/rlprocess_py27/bin/python',
    'script': '/home/vantena/scripts/record_linkage/console_febrl.py',
    'mode': 'link',
    'config': '/home/vantena/scripts/record_linkage/configs/config2.ini',
}


#################
# ANALYTICS API #
#################
PROTO_SCHEME = 'https'

ANALYTICS_API_PARAMS = {
    'module': 'API',
    'method': 'Live.getLastVisitsDetails',
    'period': 'day',
    'format': 'JSON',
}

try:
    from local_config import *
except ImportError as e:
    if "local_config" not in str(e):
        raise e


######################
# SENTIMENT ANALYSIS #
######################

DEFAULT_NLTK_DATA_FOLDER = os.path.expanduser(os.path.join('~', 'nltk_data'))
NLTK_DATA_PATH = os.environ.get('NLTK_DATA_PATH', DEFAULT_NLTK_DATA_FOLDER)
DEFAULT_ANALYSER = 'Simple'


####################
# DEBUG PARAMETERS #
####################

TEST_RUN = False
PATCH_IP_ADDRESSES = True
