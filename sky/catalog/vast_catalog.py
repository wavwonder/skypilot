""" Vast | Catalog

This module loads the service catalog file and can be used to
query instance types and pricing information for Vast.ai.
"""

import typing
from typing import Dict, List, Optional, Tuple, Union

from sky.catalog import common
from sky.catalog import vast_live
from sky.utils import ux_utils

if typing.TYPE_CHECKING:
    from sky.clouds import cloud

_df = common.read_catalog('vast/vms.csv')


def _catalog():
    return vast_live.merge(_df)


def instance_type_exists(instance_type: str) -> bool:
    return common.instance_type_exists_impl(_catalog(), instance_type)


def validate_region_zone(
        region: Optional[str],
        zone: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    return common.validate_region_zone_impl('vast', _catalog(), region, zone)


def get_hourly_cost(instance_type: str,
                    use_spot: bool = False,
                    region: Optional[str] = None,
                    zone: Optional[str] = None) -> float:
    """Returns the cost, or the cheapest cost among all zones for spot."""
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    return common.get_hourly_cost_impl(_catalog(), instance_type, use_spot,
                                       region, zone)


def get_vcpus_mem_from_instance_type(
        instance_type: str) -> Tuple[Optional[float], Optional[float]]:
    return common.get_vcpus_mem_from_instance_type_impl(_catalog(),
                                                        instance_type)


def get_default_instance_type(cpus: Optional[str] = None,
                              memory: Optional[str] = None,
                              disk_tier: Optional[str] = None,
                              region: Optional[str] = None,
                              zone: Optional[str] = None,
                              disk_size: int = 20,
                              use_spot: bool = False) -> Optional[str]:
    del disk_tier
    if zone is not None:
        raise ValueError('Vast does not support zones.')
    return vast_live.select(cpus, memory, disk_size, use_spot, region)


def get_accelerators_from_instance_type(
        instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
    return common.get_accelerators_from_instance_type_impl(
        _catalog(), instance_type)


def get_instance_type_for_accelerator(
        acc_name: str,
        acc_count: int,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        use_spot: bool = False,
        region: Optional[str] = None,
        zone: Optional[str] = None) -> Tuple[Optional[List[str]], List[str]]:
    """Returns a list of instance types that have the given accelerator."""
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    return common.get_instance_type_for_accelerator_impl(df=_catalog(),
                                                         acc_name=acc_name,
                                                         acc_count=acc_count,
                                                         cpus=cpus,
                                                         memory=memory,
                                                         use_spot=use_spot,
                                                         region=region,
                                                         zone=zone)


def get_region_zones_for_instance_type(instance_type: str,
                                       use_spot: bool) -> List['cloud.Region']:
    df = _catalog()
    df = df[df['InstanceType'] == instance_type]
    return common.get_region_zones(df, use_spot)


# TODO: this differs from the fluffy catalog version
def list_accelerators(
        gpus_only: bool,
        name_filter: Optional[str],
        region_filter: Optional[str],
        quantity_filter: Optional[int],
        case_sensitive: bool = True,
        all_regions: bool = False,
        require_price: bool = True) -> Dict[str, List[common.InstanceTypeInfo]]:
    """Returns all instance types in Vast offering GPUs."""
    del require_price  # Unused.
    return common.list_accelerators_impl('Vast', _catalog(), gpus_only,
                                         name_filter, region_filter,
                                         quantity_filter, case_sensitive,
                                         all_regions)
