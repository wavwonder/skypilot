from unittest import mock

import pytest

from sky.provision import common
from sky.provision.vast import instance
from sky.provision.vast import utils as vast_utils
from sky.utils import status_lib


def _vast_record(**overrides):
    record = {
        'id': '60927564',
        'name': 'cluster-head',
        'status': 'RUNNING',
        'local_ipaddrs': '192.168.1.3 172.17.0.1',
        'public_ipaddr': '38.114.93.61',
        'ports': {
            '22/tcp': [{
                'HostIp': '0.0.0.0',
                'HostPort': '21758',
            }]
        },
        'ssh_host': 'ssh9.vast.ai',
        'ssh_port': 22408,
    }
    record.update(overrides)
    return record


def test_get_cluster_info_prefers_ready_direct_endpoint() -> None:
    record = _vast_record()
    with (mock.patch.object(instance,
                            '_filter_instances',
                            return_value={'60927564': record}),
          mock.patch.object(instance,
                            '_probe_ssh_endpoint',
                            return_value='ready') as probe):
        info = instance.get_cluster_info('US', 'cluster')

    head = info.get_head_instance()
    assert head is not None
    assert (head.external_ip, head.ssh_port) == ('38.114.93.61', 21758)
    assert head.node_name == '60927564'
    probe.assert_called_once_with('38.114.93.61', 21758)


def test_get_cluster_info_retries_transient_direct_refusal() -> None:
    record = _vast_record()
    with (mock.patch.object(instance,
                            '_filter_instances',
                            return_value={'60927564': record}),
          mock.patch.object(instance,
                            '_probe_ssh_endpoint',
                            side_effect=['refused', 'refused', 'ready']),
          mock.patch.object(instance.time, 'sleep') as sleep_mock):
        info = instance.get_cluster_info('US', 'cluster')

    head = info.get_head_instance()
    assert head is not None
    assert (head.external_ip, head.ssh_port) == ('38.114.93.61', 21758)
    sleep_mock.assert_called_once_with(
        instance._DIRECT_SSH_PROBE_RETRY_INTERVAL_SECONDS)


def test_get_cluster_info_falls_back_when_direct_unreachable() -> None:
    record = _vast_record()
    with (mock.patch.object(instance,
                            '_filter_instances',
                            return_value={'60927564': record}),
          mock.patch.object(instance,
                            '_probe_ssh_endpoint',
                            return_value='unreachable') as probe):
        info = instance.get_cluster_info('US', 'cluster')

    head = info.get_head_instance()
    assert head is not None
    assert (head.external_ip, head.ssh_port) == ('ssh9.vast.ai', 22408)
    probe.assert_called_once_with('38.114.93.61', 21758)


def test_get_cluster_info_falls_back_when_proxy_ready() -> None:
    record = _vast_record()
    with (mock.patch.object(instance,
                            '_filter_instances',
                            return_value={'60927564': record}),
          mock.patch.object(instance,
                            '_probe_ssh_endpoint',
                            side_effect=['refused', 'ready']),
          mock.patch.object(instance.time, 'sleep') as sleep_mock):
        info = instance.get_cluster_info('US', 'cluster')

    head = info.get_head_instance()
    assert head is not None
    assert (head.external_ip, head.ssh_port) == ('ssh9.vast.ai', 22408)
    sleep_mock.assert_not_called()


def test_get_cluster_info_uses_proxy_without_probing_when_no_direct() -> None:
    record = _vast_record(ports={})
    with (mock.patch.object(instance,
                            '_filter_instances',
                            return_value={'60927564': record}),
          mock.patch.object(instance, '_probe_ssh_endpoint') as probe):
        info = instance.get_cluster_info('US', 'cluster')

    head = info.get_head_instance()
    assert head is not None
    assert (head.external_ip, head.ssh_port) == ('ssh9.vast.ai', 22408)
    probe.assert_not_called()


def test_get_cluster_info_uses_direct_without_probing_when_no_proxy() -> None:
    record = _vast_record(ssh_host=None, ssh_port=None)
    with (mock.patch.object(instance,
                            '_filter_instances',
                            return_value={'60927564': record}),
          mock.patch.object(instance, '_probe_ssh_endpoint') as probe):
        info = instance.get_cluster_info('US', 'cluster')

    head = info.get_head_instance()
    assert head is not None
    assert (head.external_ip, head.ssh_port) == ('38.114.93.61', 21758)
    probe.assert_not_called()


def test_get_cluster_info_raises_when_no_ssh_endpoint_exists() -> None:
    record = _vast_record(ports={}, ssh_host=None, ssh_port=None)
    with (mock.patch.object(instance,
                            '_filter_instances',
                            return_value={'60927564': record}),
          pytest.raises(RuntimeError, match='No SSH endpoint found')):
        instance.get_cluster_info('US', 'cluster')


def test_probe_ssh_endpoint_distinguishes_states() -> None:
    sock_ready = mock.MagicMock()
    sock_ready.__enter__.return_value = sock_ready
    sock_ready.recv.return_value = b'SSH-2.0-OpenSSH_9.2p1\r\n'
    with mock.patch.object(instance.socket,
                           'create_connection',
                           return_value=sock_ready):
        assert instance._probe_ssh_endpoint('38.114.93.61', 21758) == 'ready'

    sock_eof = mock.MagicMock()
    sock_eof.__enter__.return_value = sock_eof
    sock_eof.recv.return_value = b''
    with mock.patch.object(instance.socket,
                           'create_connection',
                           return_value=sock_eof):
        assert instance._probe_ssh_endpoint('38.114.93.61', 21758) == 'refused'

    with mock.patch.object(instance.socket,
                           'create_connection',
                           side_effect=ConnectionRefusedError):
        assert instance._probe_ssh_endpoint('38.114.93.61', 21758) == 'refused'

    with mock.patch.object(instance.socket,
                           'create_connection',
                           side_effect=TimeoutError):
        assert (instance._probe_ssh_endpoint('38.114.93.61',
                                             21758) == 'unreachable')


def test_list_instances_maps_none_actual_status_to_created() -> None:
    sdk = mock.MagicMock()
    sdk.show_instances.return_value = [
        {
            'id': 123,
            'actual_status': None,
            'label': 'c-head'
        },
        {
            'id': 456,
            'actual_status': 'running',
            'label': 'c-worker'
        },
    ]
    with mock.patch.object(vast_utils.vast, 'vast', return_value=sdk):
        result = vast_utils.list_instances()

    assert result['123']['status'] == 'CREATED'
    assert result['456']['status'] == 'RUNNING'


def test_query_instances_maps_extended_statuses() -> None:
    raw = {
        '1': {
            'id': '1',
            'name': 'c-head',
            'status': 'CREATED'
        },
        '2': {
            'id': '2',
            'name': 'c-worker',
            'status': 'UNKNOWN'
        },
        '3': {
            'id': '3',
            'name': 'c-worker',
            'status': 'FROZEN'
        },
        '4': {
            'id': '4',
            'name': 'c-worker',
            'status': 'RUNNING'
        },
    }
    with mock.patch.object(instance, '_filter_instances', return_value=raw):
        statuses = instance.query_instances('c', 'c', provider_config={})

    assert statuses == {
        '1': (status_lib.ClusterStatus.INIT, None),
        '2': (status_lib.ClusterStatus.STOPPED, None),
        '3': (status_lib.ClusterStatus.STOPPED, None),
        '4': (status_lib.ClusterStatus.UP, None),
    }


def test_run_instances_raises_on_failed_start_and_tracks_resumed() -> None:
    cfg = common.ProvisionConfig(
        provider_config={},
        authentication_config={},
        docker_config={},
        node_config={
            'InstanceType': '1x-RTX_4090-16-32768',
            'DiskSize': 50,
            'Preemptible': False,
            'ImageId': 'vastai/base:0.0.2',
        },
        count=1,
        tags={},
        resume_stopped_nodes=True,
        ports_to_open_on_launch=None,
    )
    stopped = {
        '99': {
            'id': '99',
            'name': 'c-head',
            'status': 'STOPPED',
            'ssh_port': None
        }
    }
    restarting = {
        '99': {
            'id': '99',
            'name': 'c-head',
            'status': 'STOPPED',
            'ssh_port': None
        }
    }
    running = {
        '99': {
            'id': '99',
            'name': 'c-head',
            'status': 'RUNNING',
            'ssh_port': 2222
        }
    }
    with (mock.patch.object(instance,
                            '_filter_instances',
                            side_effect=[stopped, restarting, running]),
          mock.patch.object(instance.utils, 'start') as start_mock,
          mock.patch.object(instance.time, 'sleep')):
        rec = instance.run_instances('US', 'c', 'c', cfg)

    start_mock.assert_called_once_with('99')
    assert rec.resumed_instance_ids == ['99']
    assert rec.head_instance_id == '99'

    failed = {
        '100': {
            'id': '100',
            'name': 'c-head',
            'status': 'LOADING',
            'intended_status': 'stopped',
        }
    }
    with (mock.patch.object(instance, '_filter_instances',
                            return_value=failed),
          pytest.raises(RuntimeError, match='stopped while starting')):
        instance.run_instances('US', 'c', 'c', cfg)


def test_cluster_info_and_provision_config_redaction() -> None:
    prov_cfg = common.ProvisionConfig(
        provider_config={
            'docker_login_config': {
                'password': 'secret'
            },
            'create_instance_kwargs': {
                'login': '-p secret',
                'api_key': 'key123'
            },
        },
        authentication_config={},
        docker_config={'docker_login_config': {
            'password': 'secret'
        }},
        node_config={},
        count=1,
        tags={},
        resume_stopped_nodes=False,
        ports_to_open_on_launch=None,
    )
    redacted_prov = prov_cfg.get_redacted_config()
    assert redacted_prov['docker_config']['docker_login_config'][
        'password'] == '<redacted>'
    assert redacted_prov['provider_config']['docker_login_config'][
        'password'] == '<redacted>'
    assert redacted_prov['provider_config']['create_instance_kwargs'][
        'login'] == '<redacted>'
    assert redacted_prov['provider_config']['create_instance_kwargs'][
        'api_key'] == '<redacted>'

    cluster_info = common.ClusterInfo(
        instances={},
        head_instance_id='1',
        provider_name='vast',
        provider_config={
            'create_instance_kwargs': {
                'login': '-p secret',
                'api_key': 'key123'
            },
        },
    )
    redacted_info = cluster_info.get_redacted_config()
    assert redacted_info['provider_config']['create_instance_kwargs'][
        'login'] == '<redacted>'
    assert redacted_info['provider_config']['create_instance_kwargs'][
        'api_key'] == '<redacted>'
