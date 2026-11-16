"""Live CPU-only feasibility for Vast's marketplace-backed instance types."""

import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Optional

import filelock

from sky.adaptors import common as adaptors_common
from sky.adaptors import vast
from sky.catalog import common

pd = adaptors_common.LazyImport('pandas')


def _path() -> Path:
    return Path(common.get_catalog_path('vast/live-vms.csv'))


def merge(base):
    path = _path()
    if not path.exists():
        return base
    live = pd.read_csv(path)
    return pd.concat([live, base.copy()], ignore_index=True).drop_duplicates(
        ['InstanceType', 'Region'], keep='first')


def _save(rows):
    path = _path()
    with filelock.FileLock(str(path) + '.lock'):
        if path.exists():
            rows = pd.concat([rows, pd.read_csv(path)], ignore_index=True)
        rows = rows.drop_duplicates(['InstanceType', 'Region'], keep='first')
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / 'live-vms.csv'
            rows.to_csv(temporary, index=False)
            temporary.replace(path)


def select(
    cpus: Optional[str],
    memory: Optional[str],
    disk_size: int,
    use_spot: bool,
    region: Optional[str],
    max_hourly_cost: Optional[float] = None,
    datacenter_only: bool = False,
    override_configs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    query: Dict[str, Any] = {
        'rentable': {
            'eq': True
        },
        'rented': {
            'eq': False
        },
        'verified': {
            'eq': True
        },
        'disk_space': {
            'gte': disk_size
        },
    }
    query.update(vast.resource_query(cpus, memory))
    if region is not None:
        query.update(vast.region_query(region[-2:]))
    query.update(
        vast.policy_query(
            preemptible=use_spot,
            datacenter_only=datacenter_only,
            include_cuda=True,
            max_hourly_cost=max_hourly_cost,
            override_configs=override_configs,
        ))

    offers = vast.vast().search_offers(
        query=query,
        storage=disk_size,
        type='bid' if use_spot else 'on-demand',
        order=vast.search_order(use_spot, override_configs=override_configs),
    )
    if not isinstance(offers, list):
        raise RuntimeError('Vast live offer search failed.')
    offers = vast.filter_and_sort_offers(
        offers,
        preemptible=use_spot,
        datacenter_only=datacenter_only,
        include_cuda=True,
        max_hourly_cost=max_hourly_cost,
        memory=memory,
        override_configs=override_configs,
    )
    if not offers:
        return None

    rows = []
    for offer in offers:
        gpu = offer['gpu_name']
        count = int(offer['num_gpus'])
        cores = float(offer['cpu_cores_effective'])
        ram = int(offer['cpu_ram'])
        accelerator = re.sub(r'\s', '', gpu)
        gpu_memory = offer.get('gpu_ram', 0)
        total = float(offer['dph_total'])
        spot_price = vast.offer_hourly_price(offer, preemptible=True)
        hosting_type = (1 if offer.get('datacenter') else int(
            offer.get('hosting_type', 0) or 0))
        rows.append({
            'InstanceType': f'{count}x-{gpu.replace(" ", "_")}-{cores}-{ram}',
            'AcceleratorName': accelerator,
            'AcceleratorCount': count,
            'vCPUs': cores,
            'MemoryGiB': ram / 1024,
            'Price': total,
            'SpotPrice': spot_price,
            'Region': offer['geolocation'],
            'HostingType': hosting_type,
            'GpuInfo': json.dumps({
                'Gpus': [{
                    'Name': accelerator,
                    'Count': count,
                    'MemoryInfo': {
                        'SizeInMiB': gpu_memory
                    }
                }],
                'TotalGpuMemoryInMiB': gpu_memory * count,
            }),
        })
    frame = pd.DataFrame(rows)
    frame = common._filter_with_cpus(frame, cpus)  # pylint: disable=protected-access
    frame = common._filter_with_mem(frame, memory)  # pylint: disable=protected-access
    if frame.empty:
        return None
    _save(frame)
    return str(frame['InstanceType'].iloc[0])
