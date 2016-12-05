import json
import urllib
import urllib3
import IP2Location
import itertools

from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from urllib3.exceptions import RequestError, MaxRetryError
from sqlalchemy import Table, MetaData, select
from geopy.geocoders import Nominatim

from settings import PROTO_SCHEME, API_DOMAIN, ANALYTICS_API_PARAMS, LOCATION_DB_PATH
from ..utils import validate_ip, shallow_flatten
from ..sources.connection import DBConnection
from ..exceptions.exc import IpInfoError


class WebClientAuth(DBConnection):
    """ Class that creates a connection between the
        piwik analytics tables.
    """
    def __init__(self, **kwargs):
        self.db = kwargs.get('db')
        self.subdomain = kwargs.get('subdomain')
        super(WebClientAuth, self).__init__(database=self.db)

    def get_client(self):
        """ returns client `token_auth` and `idsite`
        """
        # piwik_user table
        piwik_user = Table('vsweb_user', MetaData(bind=self.cursor), autoload=True)
        # retrieve token_auth from the piwik_user table
        # clause : piwik_user.login == subdomain
        token_auth = list(self.cursor.execute(
           select([piwik_user.c.token_auth]).where(piwik_user.c.login==self.subdomain)))[0][0]

        # get idsite
        # for now we will try to only get the first site that
        # the client admin has added.
        piwik_site = Table('vsweb_site', MetaData(bind=self.cursor), autoload=True)
        # retrieve idSite from the piwik_site table.
        site_id = list(self.cursor.execute(select([piwik_site.c.idsite])))[0][0] #get the first

        return token_auth, site_id

    def get_client1(self):
        # FOR TESTING PURPOSES
        return '1f746734a5b62f061870d66a987917f1', 1


class VSWebRequest(WebClientAuth):
    """ Class that creates a connection to the
        Web Analytics API endpoints. (Piwik)
    """
    def __init__(self, **kwargs):
        self.subdomain = kwargs.get('subdomain')
        super(VSWebRequest, self).__init__(**kwargs)

    def build_uri(self):
        """ build the api endpoint based on the client's
            subdomain.
        """
        # connects to the analytics tables and find the token_auth
        # and site_id based on the given subdomain
        token_auth, site_id = self.get_client()
        
        # API parameters
        params = ANALYTICS_API_PARAMS
        params.update({
            'idSite': site_id,
            'token_auth': token_auth,
        })

        return "{proto_scheme}://{subdomain}.{domain}/analytics/?{params}".format(
            proto_scheme=PROTO_SCHEME,
            subdomain=self.subdomain,
            domain=API_DOMAIN,
            params=urllib.parse.urlencode({
                'module': 'API',
                'method': 'Live.getLastVisitsDetails',
                'idSite': site_id,
                'period': 'day',
                'format': 'JSON',
                'token_auth': token_auth,
            }),
        )

    @property
    def data(self):
        """ get the json data from the remote host.
        """
        http = urllib3.PoolManager(timeout=5.0)
        try:
            # get data from the analytics api
            response = http.request('GET', self.build_uri())
            data = json.loads(response.data.decode('utf8'))

            rows = [shallow_flatten(item) for item in data]
            flatten = list(itertools.chain(*rows))

            # log
            print ("Data has been retrieved from the API")
            return flatten

        except (RequestError, MaxRetryError, ValueError) as e:
            raise e


class GeoInformation(object):
    """ Class that evaluates the IP address from
        the list of data and generates the coordinates
        and text address.
    """
    def __init__(self, **kwargs):
        self.location_db = IP2Location.IP2Location()
        self.location_db.open(LOCATION_DB_PATH) #open location db
        super(GeoInformation, self).__init__()

    def get_geoinformation(self, ip_address):
        """ returns a dict containing the information
            gathered based on the provided IP address.
        """
        location = {'Domain':'', 'ip':'', 'web_city':'', 'Isp':'',
            'web_country':'', 'Longitude_derived':'', 'Latitude_derived':''}
        if validate_ip(ip_address):
            record = self.location_db.get_all(ip_address)

            if record:
                web_address = self._get_web_address(record.latitude, record.longitude)
                #web_address = ''
                location.update({
                    'Domain': record.domain or '',
                    'ip': ip_address or '',
                    'Isp': record.isp or '',
                    'web_country': record.country_long or '',
                    'web_city': record.city or '',
                    'Longitude_derived': record.longitude or '',
                    'Latitude_derived': record.latitude or '',
                    'web_address': web_address,
                })
                return location

        # TODO: add logging here
        print (str(IpInfoError('Bad IP address: {}'.format(ip_address))))
        return location #no values

    def _get_web_address(self, latitude, longitude):
        """ return the text address based on the
            provided coordinates.
        """
        try:
            geolocator = Nominatim()
            location = geolocator.reverse("{latitude}, {longitude}".format(
                latitude=latitude, longitude=longitude), timeout=None)
            return location.address

        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print("Error: geocode failed: %s" % e)
            return ""

    def get_long_lat(self, address):
        try:
            geolocator = Nominatim()
            location = geolocator.geocode(address)
            return location.latitude, location.longitude

        except Exception as e:
            print(e)
            return None, None