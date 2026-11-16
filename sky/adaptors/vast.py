"""Vast cloud adaptor."""

import functools
from typing import Any, Dict, List, Optional

from sky import skypilot_config

_vast_sdk = None

POLICY_KEYS = (
    'datacenter_only',
    'min_cuda_version',
    'min_reliability',
    'min_inet_down',
    'min_inet_up',
    'min_direct_port_count',
    'min_duration_hours',
    'static_ip_only',
    'max_hourly_price',
    'order_by',
    'bid_multiplier',
    'exclude_machine_ids',
    'exclude_host_ids',
)


def get_policy(
    policy_overrides: Optional[Dict[str, Any]] = None,
    override_configs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve Vast offer-selection policy keys from SkyPilot config."""
    policy: Dict[str, Any] = {}
    for key in POLICY_KEYS:
        val = skypilot_config.get_nested(
            ('vast', key), None, override_configs=override_configs)
        if val is not None:
            policy[key] = val
    if policy_overrides:
        for key in POLICY_KEYS:
            if policy_overrides.get(key) is not None:
                policy[key] = policy_overrides[key]
    return policy


def _effective_max_hourly_price(
    policy: Dict[str, Any],
    max_hourly_cost: Optional[float] = None,
) -> Optional[float]:
    limits = [
        float(v) for v in (policy.get('max_hourly_price'), max_hourly_cost)
        if v is not None
    ]
    return min(limits) if limits else None


def offer_hourly_price(offer: Dict[str, Any],
                       preemptible: bool = False) -> float:
    """Return effective hourly price (including allocated disk) for an offer."""
    total = offer.get('dph_total')
    if not preemptible:
        return float(total if total is not None else offer.get('min_bid', 0.0))
    if 'dph_base' in offer and total is not None:
        disk_cost = max(0.0, float(total) - float(offer['dph_base']))
    else:
        disk_cost = 0.0
    base_bid = offer.get('min_bid')
    if base_bid is None:
        return float(total if total is not None else 0.0)
    return float(base_bid) + disk_cost


def compute_bid_price(
    offer: Dict[str, Any],
    policy_overrides: Optional[Dict[str, Any]] = None,
    override_configs: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Compute spot bid price applying bid_multiplier and max_hourly_price."""
    base_bid = offer.get('min_bid')
    if base_bid is None:
        return None
    policy = get_policy(policy_overrides=policy_overrides,
                        override_configs=override_configs)
    multiplier = policy.get('bid_multiplier')
    bid = float(base_bid) * (float(multiplier)
                             if multiplier is not None else 1.0)
    max_price = policy.get('max_hourly_price')
    if max_price is not None:
        bid = min(bid, float(max_price))
    return bid


def search_order(
    preemptible: bool = False,
    policy_overrides: Optional[Dict[str, Any]] = None,
    override_configs: Optional[Dict[str, Any]] = None,
) -> str:
    """Map Vast order_by policy to Vast search_offers order string."""
    policy = get_policy(policy_overrides=policy_overrides,
                        override_configs=override_configs)
    order_by = policy.get('order_by') or 'price'
    if order_by == 'score':
        return 'score-'
    if order_by == 'reliability':
        return 'reliability-'
    return 'min_bid' if preemptible else 'dph_total'


def resource_query(cpus: Optional[str] = None,
                   memory: Optional[str] = None) -> Dict[str, Any]:
    """Build structured Vast query clauses for CPU and RAM requirements."""
    query: Dict[str, Any] = {}
    for field, value, scale in (('cpu_cores_effective', cpus, 1),
                                ('cpu_ram', memory, 1024)):
        if value is None or str(value).endswith('x'):
            continue
        text = str(value)
        query[field] = {
            'gte' if text.endswith('+') else 'eq':
                float(text.rstrip('+')) * scale
        }
    return query


def policy_query(
    *,
    preemptible: bool = False,
    datacenter_only: Optional[bool] = None,
    include_cuda: bool = True,
    max_hourly_cost: Optional[float] = None,
    policy_overrides: Optional[Dict[str, Any]] = None,
    override_configs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build structured Vast search_offers query clauses from Vast policies."""
    policy = get_policy(policy_overrides=policy_overrides,
                        override_configs=override_configs)
    query: Dict[str, Any] = {}

    if datacenter_only or policy.get('datacenter_only'):
        query['datacenter'] = {'eq': True}
    if include_cuda and policy.get('min_cuda_version') is not None:
        query['cuda_max_good'] = {'gte': float(policy['min_cuda_version'])}
    if policy.get('min_reliability') is not None:
        query['reliability'] = {'gte': float(policy['min_reliability'])}
    if policy.get('min_inet_down') is not None:
        query['inet_down'] = {'gte': float(policy['min_inet_down'])}
    if policy.get('min_inet_up') is not None:
        query['inet_up'] = {'gte': float(policy['min_inet_up'])}
    if policy.get('min_direct_port_count') is not None:
        query['direct_port_count'] = {
            'gte': int(policy['min_direct_port_count'])
        }
    if policy.get('min_duration_hours') is not None:
        query['duration'] = {
            'gte': float(policy['min_duration_hours']) * 3600.0
        }
    if policy.get('static_ip_only'):
        query['static_ip'] = {'eq': True}

    max_price = _effective_max_hourly_price(policy, max_hourly_cost)
    if max_price is not None:
        query['min_bid' if preemptible else 'dph_total'] = {'lte': max_price}

    if policy.get('exclude_machine_ids'):
        query['machine_id'] = {
            'notin': [int(x) for x in policy['exclude_machine_ids']]
        }
    if policy.get('exclude_host_ids'):
        query['host_id'] = {
            'notin': [int(x) for x in policy['exclude_host_ids']]
        }
    return query


def _offer_matches_policy(
    offer: Dict[str, Any],
    *,
    policy: Dict[str, Any],
    preemptible: bool,
    datacenter_only: bool,
    include_cuda: bool,
    max_price: Optional[float],
    memory_ratio: Optional[float],
    excluded_machines: set,
    excluded_hosts: set,
) -> bool:
    if datacenter_only:
        is_dc = offer.get('datacenter')
        if is_dc is not None:
            if not is_dc:
                return False
        elif (offer.get('hosting_type') or 0) < 1:
            return False

    if include_cuda and policy.get('min_cuda_version') is not None:
        cuda_val = offer.get('cuda_max_good')
        if cuda_val is None or float(cuda_val) < float(
                policy['min_cuda_version']):
            return False

    numeric_mins = (
        ('reliability', 'min_reliability', float),
        ('inet_down', 'min_inet_down', float),
        ('inet_up', 'min_inet_up', float),
        ('direct_port_count', 'min_direct_port_count', int),
    )
    for offer_key, policy_key, cast in numeric_mins:
        threshold = policy.get(policy_key)
        if threshold is not None:
            val = offer.get(offer_key)
            if val is None or cast(val) < cast(threshold):
                return False

    if policy.get('min_duration_hours') is not None:
        duration = offer.get('duration')
        if duration is None or float(duration) < float(
                policy['min_duration_hours']) * 3600.0:
            return False

    if policy.get('static_ip_only') and not offer.get('static_ip'):
        return False

    if max_price is not None and (offer_hourly_price(offer, preemptible) >
                                  max_price + 1e-9):
        return False

    mid = offer.get('machine_id')
    if mid is not None and int(mid) in excluded_machines:
        return False

    hid = offer.get('host_id')
    if hid is not None and int(hid) in excluded_hosts:
        return False

    if memory_ratio is not None:
        ram = offer.get('cpu_ram')
        cores = offer.get('cpu_cores_effective')
        if (ram is None or cores is None or
                float(ram) < float(cores) * memory_ratio * 1024):
            return False

    return True


def filter_and_sort_offers(
    offers: List[Dict[str, Any]],
    *,
    preemptible: bool = False,
    datacenter_only: Optional[bool] = None,
    include_cuda: bool = True,
    max_hourly_cost: Optional[float] = None,
    memory: Optional[str] = None,
    policy_overrides: Optional[Dict[str, Any]] = None,
    override_configs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Filter and sort raw Vast offers according to configured policies."""
    policy = get_policy(policy_overrides=policy_overrides,
                        override_configs=override_configs)
    use_datacenter = bool(datacenter_only or policy.get('datacenter_only'))
    max_price = _effective_max_hourly_price(policy, max_hourly_cost)
    excluded_machines = {
        int(x) for x in (policy.get('exclude_machine_ids') or [])
    }
    excluded_hosts = {int(x) for x in (policy.get('exclude_host_ids') or [])}
    memory_ratio = (float(str(memory)[:-1])
                    if memory is not None and str(memory).endswith('x') else
                    None)

    filtered = [
        offer for offer in offers
        if _offer_matches_policy(
            offer,
            policy=policy,
            preemptible=preemptible,
            datacenter_only=use_datacenter,
            include_cuda=include_cuda,
            max_price=max_price,
            memory_ratio=memory_ratio,
            excluded_machines=excluded_machines,
            excluded_hosts=excluded_hosts,
        )
    ]

    order_by = policy.get('order_by') or 'price'
    if order_by == 'score':
        filtered.sort(key=lambda o: (-float(o.get('score') or 0.0),
                                     offer_hourly_price(o, preemptible)))
    elif order_by == 'reliability':
        filtered.sort(key=lambda o: (-float(o.get('reliability') or 0.0),
                                     offer_hourly_price(o, preemptible)))
    else:
        filtered.sort(key=lambda o: offer_hourly_price(o, preemptible))
    return filtered


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
def region_query(region: str) -> Dict[str, Any]:
    """Expand a catalog region with the SDK's geographic query helpers."""
    # pylint: disable=import-outside-toplevel
    from vastai.api.query import parse_query
    from vastai.utils import preprocess_search_query
    _, _, query = preprocess_search_query(
        f'georegion=true geolocation={region}')
    return parse_query(query)
