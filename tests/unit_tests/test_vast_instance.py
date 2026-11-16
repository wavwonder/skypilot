from unittest import mock

import pytest

from sky.provision.vast import instance


def _vast_record(**overrides):
    record = {
        'id': '60927564',
        'name': 'cluster-head',
        'status': 'RUNNING',
        'local_ipaddrs': '192.168.1.3 172.17.0.1',
        'public_ipaddr': '38.114.93.61',
        'ports': {'22/tcp': [{
            'HostIp': '0.0.0.0',
            'HostPort': '21758'
        }]},
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
        assert (
            instance._probe_ssh_endpoint('38.114.93.61', 21758)
            == 'unreachable'
        )
