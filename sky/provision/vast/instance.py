"""Vast instance provisioning."""
from pathlib import Path
import shlex
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

from sky import sky_logging
from sky.provision import common
from sky.provision import docker_utils
from sky.provision.vast import utils
from sky.utils import common_utils
from sky.utils import resources_utils
from sky.utils import status_lib
from sky.utils import ux_utils

POLL_INTERVAL = 10

logger = sky_logging.init_logger(__name__)
# a much more convenient method
status_filter = lambda machine_dict, stat_list: {
    k: v for k, v in machine_dict.items() if v['status'] in stat_list
}


def _filter_instances(cluster_name_on_cloud: str,
                      status_filters: Optional[List[str]],
                      head_only: bool = False) -> Dict[str, Any]:

    instances = utils.list_instances()
    possible_names = [f'{cluster_name_on_cloud}-head']
    if not head_only:
        possible_names.append(f'{cluster_name_on_cloud}-worker')

    filtered_instances = {}
    for instance_id, instance in instances.items():
        if (status_filters is not None and
                instance['status'] not in status_filters):
            continue
        if instance.get('name') in possible_names:
            filtered_instances[instance_id] = instance
    return filtered_instances


def _get_head_instance_id(instances: Dict[str, Any]) -> Optional[str]:
    for inst_id, inst in instances.items():
        if inst.get('name') and inst['name'].endswith('-head'):
            return inst_id
    return None


def _failed_start(instance: Dict[str, Any]) -> bool:
    return (instance['status']
            in ('EXITED', 'STOPPED', 'UNKNOWN', 'OFFLINE', 'FROZEN') or
            instance.get('intended_status') == 'stopped' or
            instance.get('next_state') == 'stopped')


def run_instances(region: str, cluster_name: str, cluster_name_on_cloud: str,
                  config: common.ProvisionConfig) -> common.ProvisionRecord:
    """Runs instances for the given cluster."""
    del cluster_name  # unused
    pending_status = ['CREATED', 'RESTARTING', 'LOADING']

    create_instance_kwargs = config.provider_config.get(
        'create_instance_kwargs', {})
    redacted_provider_config = config.get_redacted_config()['provider_config']
    logger.debug(f'provider_config: {redacted_provider_config}')
    logger.debug('create_instance_kwargs from provider_config: '
                 f'{redacted_provider_config.get("create_instance_kwargs", {})}')

    # Get SSH public key path and read the content for vast.ai key injection
    ssh_public_key_path = config.authentication_config.get('ssh_public_key')
    ssh_public_key = None
    if ssh_public_key_path:
        try:
            expanded_path = Path(ssh_public_key_path).expanduser()
            with open(expanded_path, 'r', encoding='utf-8') as f:
                ssh_public_key = f.read().strip()
            logger.debug(f'Read SSH public key from {expanded_path}')
        except OSError as e:
            logger.warning(f'Failed to read SSH public key from '
                           f'{ssh_public_key_path}: {e}')

    docker_login_config = config.docker_config.get('docker_login_config')
    login_args = None
    image_name = config.node_config['ImageId']
    if docker_login_config:
        login_config = (docker_login_config if isinstance(
            docker_login_config, docker_utils.DockerLoginConfig) else
                        docker_utils.DockerLoginConfig(**docker_login_config))
        login_args = shlex.join([
            '-u',
            login_config.username,
            '-p',
            login_config.password,
            login_config.server,
        ])
        image_name = login_config.format_image(image_name)

    created_instance_ids = []
    resumed_instance_ids = []
    instances: Dict[str, Any] = {}

    while True:
        instances = _filter_instances(cluster_name_on_cloud, None)
        pending = status_filter(instances, pending_status)
        if any(_failed_start(instance) for instance in pending.values()):
            raise RuntimeError('Vast instance stopped while starting.')
        if not pending:
            break
        logger.info(f'Waiting for {len(instances)} instances to be ready.')
        time.sleep(POLL_INTERVAL)

    running_instances = status_filter(instances, ['RUNNING'])
    head_instance_id = _get_head_instance_id(running_instances)
    stopped_instances = status_filter(instances, ['EXITED', 'STOPPED'])

    if config.resume_stopped_nodes and stopped_instances:
        for instance_id in stopped_instances:
            utils.start(instance_id)
            resumed_instance_ids.append(instance_id)
    else:
        to_start_count = config.count - (len(running_instances) +
                                         len(stopped_instances))
        if to_start_count < 0:
            raise RuntimeError(f'Cluster {cluster_name_on_cloud} already has '
                               f'{len(running_instances)} nodes,'
                               f'but {config.count} are required.')
        if to_start_count == 0:
            if head_instance_id is None:
                raise RuntimeError(
                    f'Cluster {cluster_name_on_cloud} has no head node.')
            logger.info(
                f'Cluster {cluster_name_on_cloud} already has '
                f'{len(running_instances)} nodes, no need to start more.')
            return common.ProvisionRecord(provider_name='vast',
                                          cluster_name=cluster_name_on_cloud,
                                          region=region,
                                          zone=None,
                                          head_instance_id=head_instance_id,
                                          resumed_instance_ids=[],
                                          created_instance_ids=[])

        secure_only = config.provider_config.get('secure_only', False)
        policy_config = config.provider_config.get('policy_config')
        for _ in range(to_start_count):
            node_type = 'head' if head_instance_id is None else 'worker'
            try:
                instance_id = utils.launch(
                    name=f'{cluster_name_on_cloud}-{node_type}',
                    instance_type=config.node_config['InstanceType'],
                    region=region,
                    disk_size=config.node_config['DiskSize'],
                    preemptible=config.node_config['Preemptible'],
                    image_name=image_name,
                    ports=config.ports_to_open_on_launch,
                    secure_only=secure_only,
                    private_docker_registry=docker_login_config is not None,
                    login=login_args,
                    create_instance_kwargs=create_instance_kwargs,
                    ssh_public_key=ssh_public_key,
                    cpus=config.node_config.get('CPUs'),
                    memory=config.node_config.get('Memory'),
                    policy_config=policy_config,
                )
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(f'run_instances error: {e}')
                raise
            logger.info(f'Launched instance {instance_id}.')
            created_instance_ids.append(instance_id)
            if head_instance_id is None:
                head_instance_id = instance_id

    # Wait for instances to be ready.
    waiting_for_restart = set(resumed_instance_ids)
    while True:
        instances = _filter_instances(cluster_name_on_cloud, None)
        waiting_for_restart = {
            inst_id for inst_id in waiting_for_restart
            if inst_id in instances and
            instances[inst_id]['status'] in ('EXITED', 'STOPPED')
        }
        failed = [
            inst_id for inst_id, inst in instances.items()
            if inst_id not in waiting_for_restart and _failed_start(inst)
        ]
        if failed:
            raise RuntimeError(f'Vast instances failed to start: {failed}')
        instances = status_filter(instances, ['RUNNING'])
        ready_instance_cnt = 0
        for instance_id, instance in instances.items():
            if instance.get('ssh_port') is not None:
                ready_instance_cnt += 1
        logger.info('Waiting for instances to be ready: '
                    f'({ready_instance_cnt}/{config.count}).')
        if ready_instance_cnt == config.count:
            break

        time.sleep(POLL_INTERVAL)

    head_instance_id = _get_head_instance_id(instances)
    assert head_instance_id is not None, 'head_instance_id should not be None'
    return common.ProvisionRecord(provider_name='vast',
                                  cluster_name=cluster_name_on_cloud,
                                  region=region,
                                  zone=None,
                                  head_instance_id=head_instance_id,
                                  resumed_instance_ids=resumed_instance_ids,
                                  created_instance_ids=created_instance_ids)


def wait_instances(region: str, cluster_name_on_cloud: str,
                   state: Optional[status_lib.ClusterStatus]) -> None:
    del region, cluster_name_on_cloud, state


def stop_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    return action_instances('stop', cluster_name_on_cloud, provider_config,
                            worker_only)


def terminate_instances(
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    return action_instances('remove', cluster_name_on_cloud, provider_config,
                            worker_only)


def action_instances(
    fn: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    worker_only: bool = False,
) -> None:
    """See sky/provision/__init__.py"""
    del provider_config  # unused
    instances = _filter_instances(cluster_name_on_cloud, None)
    for inst_id, inst in instances.items():
        logger.debug(f'Instance {fn} {inst_id}')
        if worker_only and inst['name'].endswith('-head'):
            continue
        try:
            getattr(utils, fn)(inst_id)
        except Exception as e:  # pylint: disable=broad-except
            with ux_utils.print_exception_no_traceback():
                raise RuntimeError(
                    f'Failed to {fn} instance {inst_id}: '
                    f'{common_utils.format_exception(e, use_bracket=False)}'
                ) from e


_DIRECT_SSH_PROBE_TIMEOUT_SECONDS = 2.0
_DIRECT_SSH_PROBE_RETRIES = 3
_DIRECT_SSH_PROBE_RETRY_INTERVAL_SECONDS = 1.0


def _direct_ssh_endpoint(
        instance_info: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    public_ip = instance_info.get('public_ipaddr')
    ports = instance_info.get('ports')
    if not public_ip or not isinstance(ports, dict):
        return None
    mappings = ports.get('22/tcp')
    if not isinstance(mappings, list):
        return None
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        raw_port = mapping.get('HostPort')
        if not raw_port:
            continue
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            continue
        if port > 0:
            return str(public_ip), port
    return None


def _proxy_ssh_endpoint(
        instance_info: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    ssh_host = instance_info.get('ssh_host')
    ssh_port = instance_info.get('ssh_port')
    if not ssh_host or ssh_port is None:
        return None
    try:
        port = int(ssh_port)
    except (TypeError, ValueError):
        return None
    if port <= 0:
        return None
    return str(ssh_host), port


def _probe_ssh_endpoint(
        host: str,
        port: int,
        timeout: float = _DIRECT_SSH_PROBE_TIMEOUT_SECONDS) -> str:
    """Probes an SSH endpoint and returns 'ready', 'refused', or 'unreachable'."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            if sock.recv(100).startswith(b'SSH'):
                return 'ready'
            return 'refused'
    except ConnectionRefusedError:
        return 'refused'
    except OSError:
        return 'unreachable'


def _resolve_ssh_endpoint(instance_info: Dict[str, Any]) -> Tuple[str, int]:
    direct = _direct_ssh_endpoint(instance_info)
    proxy = _proxy_ssh_endpoint(instance_info)
    if direct is None:
        if proxy is not None:
            return proxy
        raise RuntimeError(
            f'No SSH endpoint found for Vast instance {instance_info.get("id")}'
        )
    if proxy is None:
        return direct

    for attempt in range(_DIRECT_SSH_PROBE_RETRIES):
        status = _probe_ssh_endpoint(*direct)
        if status == 'ready':
            return direct
        if status == 'unreachable':
            logger.debug(
                f'Direct SSH {direct[0]}:{direct[1]} unreachable; '
                f'falling back to proxy {proxy[0]}:{proxy[1]}.')
            return proxy
        if attempt + 1 < _DIRECT_SSH_PROBE_RETRIES:
            if _probe_ssh_endpoint(*proxy) == 'ready':
                logger.debug(
                    f'Direct SSH {direct[0]}:{direct[1]} refused while proxy '
                    f'{proxy[0]}:{proxy[1]} is ready; using proxy.')
                return proxy
            time.sleep(_DIRECT_SSH_PROBE_RETRY_INTERVAL_SECONDS)

    logger.debug(
        f'Direct SSH {direct[0]}:{direct[1]} did not become ready; '
        f'falling back to proxy {proxy[0]}:{proxy[1]}.')
    return proxy


def get_cluster_info(
        region: str,
        cluster_name_on_cloud: str,
        provider_config: Optional[Dict[str, Any]] = None) -> common.ClusterInfo:
    del region  # unused
    running_instances = _filter_instances(cluster_name_on_cloud, ['RUNNING'])
    instances: Dict[str, List[common.InstanceInfo]] = {}
    head_instance_id = None
    for instance_id, instance_info in running_instances.items():
        ssh_host, ssh_port = _resolve_ssh_endpoint(instance_info)
        instances[instance_id] = [
            common.InstanceInfo(
                instance_id=instance_id,
                internal_ip=instance_info['local_ipaddrs'].strip(),
                external_ip=ssh_host,
                ssh_port=ssh_port,
                tags={},
                node_name=instance_id,
            )
        ]
        if instance_info['name'].endswith('-head'):
            head_instance_id = instance_id

    return common.ClusterInfo(
        instances=instances,
        head_instance_id=head_instance_id,
        provider_name='vast',
        provider_config=provider_config,
    )


def open_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    raise NotImplementedError('open_ports is not supported for Vast')


def query_instances(
    cluster_name: str,
    cluster_name_on_cloud: str,
    provider_config: Optional[Dict[str, Any]] = None,
    non_terminated_only: bool = True,
    retry_if_missing: bool = False,
) -> Dict[str, Tuple[Optional['status_lib.ClusterStatus'], Optional[str]]]:
    """See sky/provision/__init__.py"""
    del cluster_name, retry_if_missing  # unused
    assert provider_config is not None, (cluster_name_on_cloud, provider_config)
    instances = _filter_instances(cluster_name_on_cloud, None)
    # "running", "frozen", "stopped", "unknown", "loading"
    status_map = {
        'CREATED': status_lib.ClusterStatus.INIT,
        'RESTARTING': status_lib.ClusterStatus.INIT,
        'LOADING': status_lib.ClusterStatus.INIT,
        'EXITED': status_lib.ClusterStatus.STOPPED,
        'STOPPED': status_lib.ClusterStatus.STOPPED,
        'UNKNOWN': status_lib.ClusterStatus.STOPPED,
        'OFFLINE': status_lib.ClusterStatus.STOPPED,
        'FROZEN': status_lib.ClusterStatus.STOPPED,
        'RUNNING': status_lib.ClusterStatus.UP,
    }
    statuses: Dict[str, Tuple[Optional['status_lib.ClusterStatus'],
                              Optional[str]]] = {}
    for inst_id, inst in instances.items():
        status = status_map.get(inst['status'])
        if non_terminated_only and status is None:
            continue
        statuses[inst_id] = (status, None)
    return statuses


def cleanup_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    provider_config: Optional[Dict[str, Any]] = None,
) -> None:
    del cluster_name_on_cloud, ports, provider_config  # Unused.


def query_ports(
    cluster_name_on_cloud: str,
    ports: List[str],
    head_ip: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> Dict[int, List[common.Endpoint]]:
    """Returns externally-accessible endpoints for the given ports.

    Vast.ai exposes container ports via SSH reverse-proxy with a fixed mapping:
      container:22   → ssh_host:ssh_port
      container:N    → ssh_host:(ssh_port + N - 21)

    In practice, only port 8080 (the second forwarded port) is supported:
      container:8080 → ssh_host:(ssh_port + 1)

    Using the raw ssh_host + service port (e.g. :8080) fails because the
    Vast.ai SSH gateway does not relay arbitrary ports directly. The correct
    externally-accessible endpoint is ssh_host:(ssh_port+1).
    """
    del head_ip, provider_config  # Unused.
    ports_to_query = resources_utils.port_ranges_to_set(ports)

    running_instances = _filter_instances(cluster_name_on_cloud, ['RUNNING'],
                                          head_only=True)
    if not running_instances:
        return {}

    head_inst = list(running_instances.values())[0]
    ssh_host = head_inst.get('ssh_host') or head_inst.get('public_ipaddr', '')
    ssh_port = head_inst.get('ssh_port')

    if not ssh_host or ssh_port is None:
        return {}

    # Vast.ai port forward: container:8080 → ssh_host:(ssh_port+1)
    # Only port 8080 is supported via the gateway's second reverse-tunnel slot.
    result: Dict[int, List[common.Endpoint]] = {}
    for port in ports_to_query:
        if port == 8080:
            external_port = ssh_port + 1
        else:
            # For other ports, fall back to the host with the original port.
            # Note: these are unlikely to be reachable unless the user has
            # configured additional port forwarding on their Vast.ai instance.
            logger.warning(
                f'Port {port} requested but Vast.ai only natively forwards '
                f'port 8080 via ssh_host:(ssh_port+1). Port {port} may not '
                f'be reachable. Use port 8080 for service endpoints.')
            external_port = port
        result[port] = [
            common.SocketEndpoint(host=ssh_host, port=external_port)
        ]

    return result
