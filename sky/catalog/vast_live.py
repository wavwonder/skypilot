"""Live CPU-only feasibility for Vast's marketplace-backed instance types."""

import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Optional

import filelock

from sky import skypilot_config
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


def select(cpus: Optional[str], memory: Optional[str], disk_size: int,
           use_spot: bool, region: Optional[str]) -> Optional[str]:
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
    min_cuda = skypilot_config.get_nested(('vast', 'min_cuda_version'), None)
    if min_cuda is not None:
        query['cuda_max_good'] = {'gte': min_cuda}
    offers = vast.vast().search_offers(
        query=query,
        storage=disk_size,
        type='bid' if use_spot else 'on-demand',
        order='min_bid' if use_spot else 'dph_total')
    if not isinstance(offers, list):
        raise RuntimeError('Vast live offer search failed.')
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
        disk_cost = total - float(offer['dph_base'])
        rows.append({
            'InstanceType': f'{count}x-{gpu.replace(" ", "_")}-{cores}-{ram}',
            'AcceleratorName': accelerator,
            'AcceleratorCount': count,
            'vCPUs': cores,
            'MemoryGiB': ram / 1024,
            'Price': total,
            'SpotPrice': float(offer.get('min_bid', total)) + disk_cost,
            'Region': offer['geolocation'],
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
    frame = common._filter_with_cpus(frame, cpus)
    frame = common._filter_with_mem(frame, memory)
    if frame.empty:
        return None
    price = 'SpotPrice' if use_spot else 'Price'
    frame = frame.sort_values(price)
    _save(frame)
    return str(frame['InstanceType'].iloc[0])
