"""Unit tests for sky.provision.runpod.instance."""
from unittest import mock

import pytest

from sky.adaptors import runpod
from sky.provision import common
from sky.provision.runpod import instance as runpod_instance
from sky.provision.runpod import utils as runpod_utils
from sky.utils import status_lib

_CLUSTER = 'sky-test'


def _pod(pod_id, status, name=f'{_CLUSTER}-head', ssh_port=None):
    return {
        'id': pod_id,
        'name': name,
        'status': status,
        'external_ip': '1.2.3.4',
        'ssh_port': ssh_port,
        'vcpu_count': 8,
    }


def _config(count=1):
    return common.ProvisionConfig(
        provider_config={'availability_zone': 'EU-RO-1'},
        authentication_config={},
        docker_config={},
        node_config={
            'InstanceType': '1x_RTX5090_SECURE',
            'DiskSize': 50,
            'ImageId': 'docker:runpod/pytorch',
            'PublicKey': 'ssh-ed25519 AAAA',
            'Preemptible': False,
            'BidPerGPU': 0.0,
        },
        count=count,
        tags={},
        resume_stopped_nodes=False,
        ports_to_open_on_launch=None,
    )


@pytest.fixture
def instances(monkeypatch):
    """Live pod table backing ``utils.list_instances``.

    ``utils.remove`` drops a pod from it, so tests observe the provisioner's
    effects on the account instead of a canned call sequence.
    """
    pods = {}
    monkeypatch.setattr(runpod_instance.utils, 'list_instances', lambda: pods)
    monkeypatch.setattr(runpod_instance.utils, 'remove',
                        lambda pod_id: pods.pop(pod_id))
    monkeypatch.setattr(runpod_instance.time, 'sleep', lambda _: None)
    return pods


def _launch_into(pods, pod_id, status, ssh_port=None):
    """A ``utils.launch`` stand-in that adds the new pod to ``pods``."""

    def launch(**kwargs):
        pods[pod_id] = _pod(pod_id,
                            status,
                            name=f"{kwargs['cluster_name']}-"
                            f"{kwargs['node_type']}",
                            ssh_port=ssh_port)
        return pod_id

    return mock.Mock(side_effect=launch)


class TestQueryInstances:

    @pytest.mark.parametrize('pod_status,expected', [
        ('PROVISIONING', status_lib.ClusterStatus.INIT),
        ('STARTING', status_lib.ClusterStatus.INIT),
        ('RUNNING', status_lib.ClusterStatus.UP),
        ('EXITED', status_lib.ClusterStatus.STOPPED),
        ('ERROR', status_lib.ClusterStatus.INIT),
    ])
    def test_maps_v2_status(self, instances, pod_status, expected):
        instances['p1'] = _pod('p1', pod_status)
        statuses = runpod_instance.query_instances(_CLUSTER,
                                                   _CLUSTER,
                                                   provider_config={})
        status, reason = statuses['p1']
        assert status == expected
        assert (reason is not None) == (pod_status == 'ERROR')

    def test_terminated_pod_is_hidden_unless_asked(self, instances):
        instances['p1'] = _pod('p1', 'TERMINATED')
        assert not runpod_instance.query_instances(
            _CLUSTER, _CLUSTER, provider_config={})
        statuses = runpod_instance.query_instances(_CLUSTER,
                                                   _CLUSTER,
                                                   provider_config={},
                                                   non_terminated_only=False)
        assert statuses == {'p1': (None, None)}

    def test_ignores_other_clusters(self, instances):
        instances['p1'] = _pod('p1', 'RUNNING', name='other-head')
        assert not runpod_instance.query_instances(
            _CLUSTER, _CLUSTER, provider_config={})


class TestRunInstances:

    def test_raises_when_launched_pod_errors(self, instances, monkeypatch):
        monkeypatch.setattr(runpod_instance.utils, 'launch',
                            _launch_into(instances, 'p1', 'ERROR'))
        with pytest.raises(RuntimeError, match='ERROR'):
            runpod_instance.run_instances('RO', _CLUSTER, _CLUSTER, _config())

    def test_deletes_stale_pods_before_launching(self, instances, monkeypatch):
        instances['p1'] = _pod('p1', 'EXITED')
        launch = _launch_into(instances, 'p2', 'RUNNING', ssh_port=2222)
        monkeypatch.setattr(runpod_instance.utils, 'launch', launch)
        record = runpod_instance.run_instances('RO', _CLUSTER, _CLUSTER,
                                               _config())
        assert 'p1' not in instances
        launch.assert_called_once()
        assert record.created_instance_ids == ['p2']
        assert record.head_instance_id == 'p2'

    def test_reuses_running_pod(self, instances, monkeypatch):
        launch = mock.Mock()
        monkeypatch.setattr(runpod_instance.utils, 'launch', launch)
        instances['p1'] = _pod('p1', 'RUNNING', ssh_port=2222)
        record = runpod_instance.run_instances('RO', _CLUSTER, _CLUSTER,
                                               _config())
        launch.assert_not_called()
        assert record.head_instance_id == 'p1'
        assert record.created_instance_ids == []


class TestGetClusterInfo:

    def test_internal_ip_is_the_public_ip(self, instances):
        instances['p1'] = _pod('p1', 'RUNNING', ssh_port=2222)
        info = runpod_instance.get_cluster_info('RO',
                                                _CLUSTER,
                                                provider_config={})
        node = info.instances['p1'][0]
        assert node.internal_ip == '1.2.3.4'
        assert node.external_ip == '1.2.3.4'
        assert node.ssh_port == 2222
        assert info.head_instance_id == 'p1'
        assert info.custom_ray_options == {'num-cpus': '8'}


class TestListInstancesEnrichment:

    def test_enriches_running_pod_when_list_omits_endpoints(self, monkeypatch):
        calls = []

        def fake_rest(method, path, json=None, params=None):
            del json, params
            calls.append((method, path))
            if (method, path) == ('GET', '/pods'):
                return {
                    'pods': [{
                        'id': 'pod-1',
                        'name': f'{_CLUSTER}-head',
                        'status': 'RUNNING',
                        'runtime': None,
                        'ssh': {
                            'direct': None
                        },
                    }],
                    'pagination': {
                        'nextCursor': None,
                        'hasNextPage': False,
                    },
                }
            if (method, path) == ('GET', '/pods/pod-1'):
                return {
                    'id': 'pod-1',
                    'name': f'{_CLUSTER}-head',
                    'status': 'RUNNING',
                    'ssh': {
                        'direct': {
                            'host': '1.2.3.4',
                            'port': 2222,
                        },
                    },
                    'runtime': {
                        'ports': [{
                            'private': 22,
                            'public': 2222,
                            'type': 'tcp',
                            'ip': '1.2.3.4',
                        }],
                    },
                }
            raise AssertionError(f'Unexpected call {method} {path}')

        monkeypatch.setattr(runpod, 'rest_request', fake_rest)

        instances = runpod_utils.list_instances()

        assert calls == [('GET', '/pods'), ('GET', '/pods/pod-1')]
        assert instances['pod-1']['external_ip'] == '1.2.3.4'
        assert instances['pod-1']['ssh_port'] == 2222
        assert instances['pod-1']['port2endpoint'] == {
            22: {
                'host': '1.2.3.4',
                'port': 2222,
            }
        }

    def test_enriches_running_pod_when_runtime_empty_and_no_ssh_direct(
            self, monkeypatch):
        calls = []

        def fake_rest(method, path, json=None, params=None):
            del json, params
            calls.append((method, path))
            if (method, path) == ('GET', '/pods'):
                return {
                    'pods': [{
                        'id': 'pod-1',
                        'name': f'{_CLUSTER}-head',
                        'status': 'RUNNING',
                        'runtime': {
                            'ports': []
                        },
                        'ssh': {
                            'direct': None
                        },
                    }],
                    'pagination': {
                        'nextCursor': None,
                        'hasNextPage': False,
                    },
                }
            if (method, path) == ('GET', '/pods/pod-1'):
                return {
                    'id': 'pod-1',
                    'name': f'{_CLUSTER}-head',
                    'status': 'RUNNING',
                    'ssh': {
                        'direct': {
                            'host': '5.6.7.8',
                            'port': 3333,
                        },
                    },
                    'runtime': {
                        'ports': []
                    },
                }
            raise AssertionError(f'Unexpected call {method} {path}')

        monkeypatch.setattr(runpod, 'rest_request', fake_rest)

        instances = runpod_utils.list_instances()

        assert calls == [('GET', '/pods'), ('GET', '/pods/pod-1')]
        assert instances['pod-1']['external_ip'] == '5.6.7.8'
        assert instances['pod-1']['ssh_port'] == 3333

    def test_skips_detail_call_when_endpoints_present_or_not_running(
            self, monkeypatch):
        calls = []

        def fake_rest(method, path, json=None, params=None):
            del json, params
            calls.append((method, path))
            assert (method, path) == ('GET', '/pods')
            return {
                'pods': [
                    {
                        'id': 'pod-direct',
                        'name': f'{_CLUSTER}-head',
                        'status': 'RUNNING',
                        'ssh': {
                            'direct': {
                                'host': '1.2.3.4',
                                'port': 2222,
                            },
                        },
                        'runtime': {
                            'ports': []
                        },
                    },
                    {
                        'id': 'pod-port22',
                        'name': f'{_CLUSTER}-worker',
                        'status': 'RUNNING',
                        'ssh': {
                            'direct': None
                        },
                        'runtime': {
                            'ports': [{
                                'private': 22,
                                'public': 2223,
                                'type': 'tcp',
                                'ip': '1.2.3.5',
                            }],
                        },
                    },
                    {
                        'id': 'pod-pending',
                        'name': 'other-head',
                        'status': 'PROVISIONING',
                        'runtime': None,
                        'ssh': {
                            'direct': None
                        },
                    },
                ],
                'pagination': {
                    'nextCursor': None,
                    'hasNextPage': False,
                },
            }

        monkeypatch.setattr(runpod, 'rest_request', fake_rest)

        instances = runpod_utils.list_instances()

        assert calls == [('GET', '/pods')]
        assert instances['pod-direct']['ssh_port'] == 2222
        assert instances['pod-port22']['ssh_port'] == 2223
        assert 'ssh_port' not in instances['pod-pending']

    def test_handles_404_during_enrichment_gracefully(self, monkeypatch):

        def fake_rest(method, path, json=None, params=None):
            del json, params
            if (method, path) == ('GET', '/pods'):
                return {
                    'pods': [{
                        'id': 'pod-gone',
                        'name': f'{_CLUSTER}-head',
                        'status': 'RUNNING',
                        'runtime': None,
                        'ssh': {
                            'direct': None
                        },
                    }],
                    'pagination': {
                        'nextCursor': None,
                        'hasNextPage': False,
                    },
                }
            if (method, path) == ('GET', '/pods/pod-gone'):
                raise runpod.RunPodRestError('Not Found', status_code=404)
            raise AssertionError(f'Unexpected call {method} {path}')

        monkeypatch.setattr(runpod, 'rest_request', fake_rest)

        assert runpod_utils.list_instances() == {}

    def test_falls_back_to_list_record_on_transient_rest_error(
            self, monkeypatch):

        def fake_rest(method, path, json=None, params=None):
            del json, params
            if (method, path) == ('GET', '/pods'):
                return {
                    'pods': [{
                        'id': 'pod-err',
                        'name': f'{_CLUSTER}-head',
                        'status': 'RUNNING',
                        'runtime': None,
                        'ssh': {
                            'direct': None
                        },
                    }],
                    'pagination': {
                        'nextCursor': None,
                        'hasNextPage': False,
                    },
                }
            if (method, path) == ('GET', '/pods/pod-err'):
                raise runpod.RunPodRestError('Server Error', status_code=500)
            raise AssertionError(f'Unexpected call {method} {path}')

        monkeypatch.setattr(runpod, 'rest_request', fake_rest)

        instances = runpod_utils.list_instances()
        assert instances['pod-err']['status'] == 'RUNNING'
        assert 'ssh_port' not in instances['pod-err']
