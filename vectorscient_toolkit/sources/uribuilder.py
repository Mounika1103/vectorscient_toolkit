from urllib.parse import urljoin


__all__ = ["UriBuilder"]


class UriBuilder:
    """
    Thin class for specific URLs building. Is used to create URLs for analytics
    data retrieving.
    """

    SCHEME = "http"

    def __init__(self, host: str, path: str= "analytics", **params):
        self._host = host
        self._view = path
        self._module = params.get("module", "API")
        self._method = params.get("method", None)
        self._id_site = params.get("id_site", None)
        self._period = params.get("period", "day")
        self._output_format = params.get("format", "JSON")
        self._token_auth = params.get("token_auth", None)
        self._uri = None

    @property
    def host_name(self):
        return self._host

    def build(self):
        uri = urljoin(self.SCHEME + "://" + self._host, self._view)
        params = (
            "module={_module}&"
            "method={_method}&"
            "idSite={_id_site}&"
            "period={_period}&"
            "format={_output_format}&"
            "token_auth={_token_auth}"
        ).format(**self.__dict__)
        return "{uri}?{params}".format(uri=uri, params=params)

    def module(self, name: str):
        self._module = name
        return self

    def method(self, name: str):
        self._method = name
        return self

    def id_site(self, value: int):
        self._id_site = value
        return self

    def period(self, value: str):
        self._period = value
        return self

    def output_format(self, value: str):
        self._output_format = value
        return self

    def token_auth(self, token: str):
        self._token_auth = token
        return self
