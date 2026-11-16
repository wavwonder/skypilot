import logging
from unittest import mock

import jsonschema
import pytest

from sky import skypilot_config
from sky.adaptors import vast as vast_adaptor
from sky.catalog import vast_live
from sky.provision.vast import utils as vast_utils
from sky.utils import schemas


def _sample_offers():
    return [
        {
            'id': 101,
            'machine_id': 1001,
            'host_id': 2001,
            'geolocation': 'US',
            'gpu_name': 'RTX 4090',
            'num_gpus': 1,
            'cpu_cores_effective': 16.0,
            'cpu_ram': 32768,
            'gpu_ram': 24576,
            'cuda_max_good': 12.4,
            'reliability': 0.995,
            'inet_down': 1200.0,
            'inet_up': 600.0,
            'direct_port_count': 4,
            'duration': 72 * 3600.0,
            'datacenter': True,
            'static_ip': True,
            'dph_total': 0.80,
            'dph_base': 0.70,
            'min_bid': 0.40,
            'score': 90.0,
        },
        {
            'id': 102,
            'machine_id': 1002,
            'host_id': 2002,
            'geolocation': 'US',
            'gpu_name': 'RTX 4090',
            'num_gpus': 1,
            'cpu_cores_effective': 8.0,
            'cpu_ram': 16384,
            'gpu_ram': 24576,
            'cuda_max_good': 12.2,
            'reliability': 0.96,
            'inet_down': 400.0,
            'inet_up': 200.0,
            'direct_port_count': 2,
            'duration': 24 * 3600.0,
            'datacenter': False,
            'static_ip': False,
            'dph_total': 0.50,
            'dph_base': 0.45,
            'min_bid': 0.20,
            'score': 95.0,
        },
        {
            'id': 103,
            'machine_id': 1003,
            'host_id': 2003,
            'geolocation': 'US',
            'gpu_name': 'RTX 4090',
            'num_gpus': 1,
            'cpu_cores_effective': 16.0,
            'cpu_ram': 65536,
            'gpu_ram': 24576,
            'cuda_max_good': 11.8,
            'reliability': 0.999,
            'inet_down': 800.0,
            'inet_up': 50.0,
            'direct_port_count': 0,
            'duration': 6 * 3600.0,
            'datacenter': True,
            'static_ip': True,
            'dph_total': 0.65,
            'dph_base': 0.60,
            'min_bid': 0.30,
            'score': 99.0,
        },
    ]


def test_vast_schema_accepts_all_policy_fields() -> None:
    schema = schemas.get_config_schema()
    config = {
        'vast': {
            'datacenter_only': True,
            'create_instance_kwargs': {
                'force': True
            },
            'min_cuda_version': 12.1,
            'min_reliability': 0.98,
            'min_inet_down': 500,
            'min_inet_up': 100,
            'min_direct_port_count': 2,
            'min_duration_hours': 24,
            'static_ip_only': True,
            'max_hourly_price': 1.25,
            'order_by': 'reliability',
            'bid_multiplier': 1.25,
            'exclude_machine_ids': [1002],
            'exclude_host_ids': [2003],
        }
    }
    jsonschema.validate(instance=config, schema=schema)


@pytest.mark.parametrize('vast_cfg', [
    {
        'order_by': 'fastest'
    },
    {
        'bid_multiplier': 0.9
    },
    {
        'min_reliability': 1.2
    },
    {
        'min_direct_port_count': 1.5
    },
    {
        'exclude_machine_ids': ['bad']
    },
    {
        'unknown_key': True
    },
])
def test_vast_schema_rejects_invalid_policy_values(vast_cfg) -> None:
    schema = schemas.get_config_schema()
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={'vast': vast_cfg}, schema=schema)


def test_policy_query_builds_expected_structured_clauses() -> None:
    override = {
        'vast': {
            'datacenter_only': True,
            'min_cuda_version': 12.1,
            'min_reliability': 0.98,
            'min_inet_down': 500,
            'min_inet_up': 100,
            'min_direct_port_count': 2,
            'min_duration_hours': 12,
            'static_ip_only': True,
            'max_hourly_price': 0.9,
            'exclude_machine_ids': [11, 12],
            'exclude_host_ids': [21],
        }
    }
    q = vast_adaptor.policy_query(preemptible=False, override_configs=override)
    assert q == {
        'datacenter': {
            'eq': True
        },
        'cuda_max_good': {
            'gte': 12.1
        },
        'reliability': {
            'gte': 0.98
        },
        'inet_down': {
            'gte': 500.0
        },
        'inet_up': {
            'gte': 100.0
        },
        'direct_port_count': {
            'gte': 2
        },
        'duration': {
            'gte': 12 * 3600.0
        },
        'static_ip': {
            'eq': True
        },
        'dph_total': {
            'lte': 0.9
        },
        'machine_id': {
            'notin': [11, 12]
        },
        'host_id': {
            'notin': [21]
        },
    }

    spot_q = vast_adaptor.policy_query(preemptible=True,
                                       include_cuda=False,
                                       max_hourly_cost=0.7,
                                       override_configs=override)
    assert 'cuda_max_good' not in spot_q
    assert spot_q['min_bid'] == {'lte': 0.7}


@pytest.mark.parametrize('policy,preemptible,expected_ids', [
    ({
        'min_reliability': 0.99
    }, False, [103, 101]),
    ({
        'min_inet_down': 500
    }, False, [103, 101]),
    ({
        'min_inet_up': 150
    }, False, [102, 101]),
    ({
        'min_direct_port_count': 2
    }, False, [102, 101]),
    ({
        'min_duration_hours': 24
    }, False, [102, 101]),
    ({
        'datacenter_only': True
    }, False, [103, 101]),
    ({
        'static_ip_only': True
    }, False, [103, 101]),
    ({
        'max_hourly_price': 0.65
    }, False, [102, 103]),
    ({
        'max_hourly_price': 0.35
    }, True, [102, 103]),
    ({
        'exclude_machine_ids': [1002]
    }, False, [103, 101]),
    ({
        'exclude_host_ids': [2001, 2002]
    }, False, [103]),
    ({
        'order_by': 'score'
    }, False, [103, 102, 101]),
    ({
        'order_by': 'reliability'
    }, False, [103, 101, 102]),
])
def test_filter_and_sort_offers_policies(policy, preemptible,
                                         expected_ids) -> None:
    offers = _sample_offers()
    result = vast_adaptor.filter_and_sort_offers(
        offers,
        preemptible=preemptible,
        override_configs={'vast': policy},
    )
    assert [o['id'] for o in result] == expected_ids


def test_launch_uses_structured_query_policies_bid_multiplier_and_logs(
        caplog) -> None:
    offers = _sample_offers()
    sdk = mock.MagicMock()
    sdk.client.api_key = 'vast-secret'
    sdk.search_offers.return_value = offers
    sdk.create_instance.return_value = {'success': True, 'new_contract': 98765}

    policy = {
        'min_reliability': 0.99,
        'min_inet_down': 500,
        'min_inet_up': 100,
        'min_direct_port_count': 1,
        'min_duration_hours': 48,
        'static_ip_only': True,
        'max_hourly_price': 0.55,
        'order_by': 'price',
        'bid_multiplier': 1.5,
        'exclude_machine_ids': [1003],
        'exclude_host_ids': [2002],
        'min_cuda_version': 12.0,
    }

    with (mock.patch.object(vast_utils.vast, 'vast', return_value=sdk),
          mock.patch.object(vast_utils.vast,
                            'region_query',
                            return_value={'geolocation': {
                                'eq': 'US'
                            }}),
          mock.patch.object(skypilot_config, '_global_config_context',
                            skypilot_config.ConfigContext()),
          caplog.at_level(logging.INFO)):
        contract_id = vast_utils.launch(
            name='cluster-head',
            instance_type='1x-RTX_4090-16-32768',
            region='US',
            disk_size=50,
            image_name='vastai/base:0.0.2',
            ports=[8080],
            preemptible=True,
            secure_only=True,
            create_instance_kwargs={'env': {
                'FOO': 'bar'
            }},
            ssh_public_key='ssh-ed25519 AAAA',
            policy_config=policy,
        )

    assert contract_id == '98765'
    search_call = sdk.search_offers.call_args.kwargs
    assert isinstance(search_call['query'], dict)
    assert search_call['query']['datacenter'] == {'eq': True}
    assert search_call['query']['reliability'] == {'gte': 0.99}
    assert search_call['query']['duration'] == {'gte': 48 * 3600.0}
    assert search_call['query']['machine_id'] == {'notin': [1003]}
    assert search_call['type'] == 'bid'
    assert search_call['storage'] == 50
    assert search_call['order'] == 'min_bid'

    create_kwargs = sdk.create_instance.call_args.kwargs
    assert create_kwargs['id'] == 101
    # min_bid=0.40 * 1.5 = 0.60, clamped to max_hourly_price=0.55
    assert create_kwargs['price'] == pytest.approx(0.55)
    assert create_kwargs['env'] == {'__SOURCE': 'skypilot', 'FOO': 'bar'}
    assert 'ssh-ed25519 AAAA' in create_kwargs['onstart_cmd']

    log_text = caplog.text
    for expected in ('id=101', 'machine_id=1001', 'host_id=2001',
                     'geolocation=US', 'reliability=0.995', 'inet_down=1200.0',
                     'inet_up=600.0', 'direct_port_count=4', 'dph_total=0.8'):
        assert expected in log_text


def test_launch_respects_pinned_price_in_create_instance_kwargs() -> None:
    offers = [_sample_offers()[0]]
    sdk = mock.MagicMock()
    sdk.client.api_key = 'vast-secret'
    sdk.search_offers.return_value = offers
    sdk.create_instance.return_value = {'success': True, 'new_contract': 111}

    with (mock.patch.object(vast_utils.vast, 'vast', return_value=sdk),
          mock.patch.object(vast_utils.vast, 'region_query', return_value={})):
        vast_utils.launch(
            name='c-head',
            instance_type='1x-RTX_4090-16-32768',
            region='US',
            disk_size=50,
            image_name='vastai/base:0.0.2',
            ports=None,
            preemptible=True,
            secure_only=False,
            create_instance_kwargs={'price': 0.42},
            policy_config={'bid_multiplier': 2.0},
        )

    assert sdk.create_instance.call_args.kwargs['price'] == 0.42


def test_vast_live_select_uses_shared_policy_and_order(tmp_path) -> None:
    offers = _sample_offers()
    sdk = mock.MagicMock()
    sdk.search_offers.return_value = offers
    live_csv = tmp_path / 'live-vms.csv'

    with (mock.patch.object(vast_live.vast, 'vast', return_value=sdk),
          mock.patch.object(vast_live.vast, 'region_query', return_value={}),
          mock.patch.object(vast_live, '_path', return_value=live_csv)):
        selected = vast_live.select(
            cpus='8+',
            memory='16+',
            disk_size=40,
            use_spot=False,
            region='US',
            override_configs={
                'vast': {
                    'min_reliability': 0.99,
                    'order_by': 'reliability',
                }
            },
        )

    assert selected == '1x-RTX_4090-16.0-65536'
    assert sdk.search_offers.call_args.kwargs['order'] == 'reliability-'
    assert live_csv.exists()


def test_vast_cloud_make_deploy_variables_includes_spot_cpus_and_policy(
) -> None:
    from sky import clouds
    from sky import resources as resources_lib
    from sky.utils import resources_utils

    r = resources_lib.Resources(
        cloud=clouds.Vast(),
        instance_type='1x-RTX_4090-16-32768',
        cpus='16+',
        memory='32+',
        use_spot=True,
        region='US',
        _cluster_config_overrides={
            'vast': {
                'min_reliability': 0.99,
                'datacenter_only': True,
                'bid_multiplier': 1.2,
            }
        },
    )
    with mock.patch.object(clouds.Vast,
                           'get_accelerators_from_instance_type',
                           return_value={'RTX4090': 1}):
        vars_dict = clouds.Vast().make_deploy_resources_variables(
            r,
            resources_utils.ClusterName('c', 'c'),
            clouds.Region('US'),
            None,
            1,
        )
    assert vars_dict['use_spot'] is True
    assert vars_dict['requested_cpus'] == '16+'
    assert vars_dict['requested_memory'] == '32+'
    assert vars_dict['secure_only'] is True
    assert vars_dict['policy_config'] == {
        'datacenter_only': True,
        'min_reliability': 0.99,
        'bid_multiplier': 1.2,
    }

