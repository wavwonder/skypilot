"""Vast cloud adaptor."""

import functools

_vast_sdk = None


def import_package(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _vast_sdk

        if _vast_sdk is None:
            try:
                import vastai_sdk as _vast  # pylint: disable=import-outside-toplevel
                _vast_sdk = _vast.VastAI()
            except ImportError as e:
                raise ImportError(f'Fail to import dependencies for vast: {e}\n'
                                  'Try pip install "skypilot[vast]"') from None
        return func(*args, **kwargs)

    return wrapper


@import_package
def vast():
    """Return the vast package."""
    return _vast_sdk


@import_package
def region_query(region):
    """Expand a catalog region with the SDK's geographic query helpers."""
    # pylint: disable=import-outside-toplevel
    from vastai.api.query import parse_query
    from vastai.utils import preprocess_search_query
    _, _, query = preprocess_search_query(
        f'georegion=true geolocation={region}')
    return parse_query(query)
