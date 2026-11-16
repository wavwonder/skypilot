from unittest import mock

from sky.exceptions import ClusterDoesNotExist
from sky.jobs import utils


@mock.patch('sky.core.down')
@mock.patch('sky.usage.usage_lib.messages.usage.set_internal')
def test_terminate_cluster_retry_on_value_error(mock_set_internal,
                                                mock_sky_down) -> None:
    # Set up mock to fail twice with ValueError, then succeed
    mock_sky_down.side_effect = [
        ValueError('Mock error 1'),
        ValueError('Mock error 2'),
        None,
    ]

    # Call should succeed after retries
    utils.terminate_cluster('test-cluster')

    # Verify sky.down was called 3 times
    assert mock_sky_down.call_count == 3
    mock_sky_down.assert_has_calls([
        mock.call('test-cluster'),
        mock.call('test-cluster'),
        mock.call('test-cluster'),
    ])

    # Verify usage.set_internal was called before each sky.down
    assert mock_set_internal.call_count == 3


@mock.patch('sky.core.down')
@mock.patch('sky.usage.usage_lib.messages.usage.set_internal')
def test_terminate_cluster_handles_nonexistent_cluster(mock_set_internal,
                                                       mock_sky_down) -> None:
    # Set up mock to raise ClusterDoesNotExist
    mock_sky_down.side_effect = ClusterDoesNotExist('test-cluster')

    # Call should succeed silently
    utils.terminate_cluster('test-cluster')

    # Verify sky.down was called once
    assert mock_sky_down.call_count == 1
    mock_sky_down.assert_called_once_with('test-cluster')

    # Verify usage.set_internal was called once
    assert mock_set_internal.call_count == 1


def _make_stub_controller():
    from sky.jobs import controller

    ctrl = controller.JobsController.__new__(controller.JobsController)
    ctrl._job_id = 1
    ctrl._pool = None
    ctrl._backend = mock.MagicMock()
    ctrl._backend.run_timestamp = '20260101-000000'
    ctrl._cleanup_cluster = mock.MagicMock()
    ctrl._download_log_and_stream = mock.MagicMock()
    return ctrl


def _make_stub_task():
    task = mock.MagicMock()
    task.name = 'test-task'
    task.run = 'echo hi'
    task.envs = {'SKYPILOT_TASK_ID': '1'}
    task.num_nodes = 1
    return task


def test_jobs_controller_checks_job_after_bounded_network_errors() -> None:
    from sky import exceptions
    from sky.jobs import controller
    from sky.skylet import job_lib

    ctrl = _make_stub_controller()
    task = _make_stub_task()
    strategy = mock.MagicMock()
    strategy.launch.return_value = 100.0
    strategy.max_restarts_on_errors = 0

    with mock.patch(
            'sky.jobs.controller.managed_job_state.get_latest_task_id_status',
            return_value=(None, None)), \
         mock.patch('sky.jobs.controller.managed_job_utils.event_callback_func',
                    return_value=mock.MagicMock()), \
         mock.patch('sky.jobs.controller.recovery_strategy.StrategyExecutor.make',
                    return_value=strategy), \
         mock.patch('sky.jobs.controller.managed_job_state.set_starting'), \
         mock.patch('sky.jobs.controller.managed_job_state.set_started'), \
         mock.patch('sky.jobs.controller.managed_job_state.set_succeeded'), \
         mock.patch('sky.jobs.controller.backend_utils.get_timestamp_from_run_timestamp',
                    return_value=99.0), \
         mock.patch('sky.jobs.controller.backend_utils.get_task_resources_str',
                    return_value='1xCPU'), \
         mock.patch('sky.jobs.controller.backend_utils.check_network_connection',
                    side_effect=exceptions.NetworkError('down')), \
         mock.patch('sky.jobs.controller.managed_job_utils.get_job_status',
                    return_value=job_lib.JobStatus.SUCCEEDED) as mock_get_status, \
         mock.patch('sky.jobs.controller.managed_job_utils.try_to_get_job_end_time',
                    return_value=200.0), \
         mock.patch('sky.jobs.controller.backend_utils.get_clusters',
                    return_value=[]), \
         mock.patch('sky.jobs.controller.time.sleep'):
        result = ctrl._run_one_task(0, task)

    assert result is True
    assert mock_get_status.call_count == 1
    ctrl._cleanup_cluster.assert_called_once()


def test_jobs_controller_retries_on_cluster_status_fetching_error() -> None:
    from sky import exceptions
    from sky.skylet import job_lib

    ctrl = _make_stub_controller()
    task = _make_stub_task()
    strategy = mock.MagicMock()
    strategy.launch.return_value = 100.0
    strategy.max_restarts_on_errors = 0

    with mock.patch(
            'sky.jobs.controller.managed_job_state.get_latest_task_id_status',
            return_value=(None, None)), \
         mock.patch('sky.jobs.controller.managed_job_utils.event_callback_func',
                    return_value=mock.MagicMock()), \
         mock.patch('sky.jobs.controller.recovery_strategy.StrategyExecutor.make',
                    return_value=strategy), \
         mock.patch('sky.jobs.controller.managed_job_state.set_starting'), \
         mock.patch('sky.jobs.controller.managed_job_state.set_started'), \
         mock.patch('sky.jobs.controller.managed_job_state.set_succeeded'), \
         mock.patch('sky.jobs.controller.backend_utils.get_timestamp_from_run_timestamp',
                    return_value=99.0), \
         mock.patch('sky.jobs.controller.backend_utils.get_task_resources_str',
                    return_value='1xCPU'), \
         mock.patch('sky.jobs.controller.backend_utils.check_network_connection',
                    return_value=None), \
         mock.patch('sky.jobs.controller.managed_job_utils.get_job_status',
                    side_effect=[None, job_lib.JobStatus.SUCCEEDED]), \
         mock.patch('sky.jobs.controller.backend_utils.refresh_cluster_status_handle',
                    side_effect=exceptions.ClusterStatusFetchingError('api down')), \
         mock.patch('sky.jobs.controller.managed_job_utils.try_to_get_job_end_time',
                    return_value=200.0), \
         mock.patch('sky.jobs.controller.backend_utils.get_clusters',
                    return_value=[]), \
         mock.patch('sky.jobs.controller.time.sleep'):
        result = ctrl._run_one_task(0, task)

    assert result is True
    strategy.recover.assert_not_called()
    ctrl._cleanup_cluster.assert_called_once()

