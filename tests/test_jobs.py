from argparse import Namespace
from pathlib import Path

import pytest

from jobs.cli import (
    WORKLOAD_CLASSES,
    SubmittedJob,
    _cancel_stalled_job_for_fallback,
    _job_safeguards,
    _job_target,
    _matrix_plan,
    _parse_environment,
    _regional_stack_name,
    _sanitize_name,
)
from jobs.ecr_replication import ensure_replication_rule
from jobs.runtime import S3Uri, _prepare_command


def test_sanitize_batch_name():
    assert _sanitize_name("GPU smoke test / seed 44") == "GPU-smoke-test-seed-44"


def test_parse_environment_rejects_credentials():
    with pytest.raises(ValueError, match="temporary AWS credentials"):
        _parse_environment(["AWS_SECRET_ACCESS_KEY=not-allowed"])


def test_high_memory_jobs_default_to_one_attempt_and_two_hour_timeout():
    args = Namespace(attempts=None, timeout_seconds=None)

    assert _job_safeguards(args, WORKLOAD_CLASSES["high_memory_gpu"]) == {
        "retryStrategy": {"attempts": 1},
        "timeout": {"attemptDurationSeconds": 7200},
    }


def test_small_gpu_jobs_keep_job_definition_safeguards_by_default():
    args = Namespace(attempts=None, timeout_seconds=None)

    assert _job_safeguards(args, WORKLOAD_CLASSES["small_gpu"]) == {}


def test_spot_targets_the_dedicated_small_gpu_queue():
    config = Namespace(
        spot_job_queue_arn="spot-queue",
        spot_max_vcpus=4,
    )

    assert _job_target(config, WORKLOAD_CLASSES["small_gpu"], "spot") == ("spot-queue", 4)


def test_regional_stack_defaults_preserve_explicit_overrides():
    assert _regional_stack_name("us-east-1") == "GeneralsTrainingBatch"
    assert _regional_stack_name("us-west-2") == "GeneralsTrainingBatchUsWest2"
    assert _regional_stack_name("us-west-2", "CustomStack") == "CustomStack"


def test_ecr_replication_rule_preserves_unrelated_registry_rules():
    existing_rule = {
        "destinations": [{"region": "us-east-2", "registryId": "111111111111"}],
        "repositoryFilters": [{"filter": "other-", "filterType": "PREFIX_MATCH"}],
    }

    class FakeEcr:
        updated = None

        def describe_registry(self):
            return {
                "registryId": "111111111111",
                "replicationConfiguration": {"rules": [existing_rule]},
            }

        def put_replication_configuration(self, *, replicationConfiguration):
            self.updated = replicationConfiguration

    ecr = FakeEcr()
    assert ensure_replication_rule(
        ecr,
        destination_region="us-west-2",
        repository_name="generals-training",
    )
    assert ecr.updated["rules"][0] == existing_rule
    assert ecr.updated["rules"][1] == {
        "destinations": [{"region": "us-west-2", "registryId": "111111111111"}],
        "repositoryFilters": [
            {"filter": "generals-training", "filterType": "PREFIX_MATCH"}
        ],
    }


def test_ecr_replication_rule_reuses_a_broader_existing_filter():
    class FakeEcr:
        def describe_registry(self):
            return {
                "registryId": "111111111111",
                "replicationConfiguration": {
                    "rules": [
                        {
                            "destinations": [
                                {"region": "us-west-2", "registryId": "111111111111"}
                            ],
                            "repositoryFilters": [
                                {"filter": "generals-", "filterType": "PREFIX_MATCH"}
                            ],
                        }
                    ]
                },
            }

        def put_replication_configuration(self, **_kwargs):
            raise AssertionError("existing replication coverage should be reused")

    assert not ensure_replication_rule(
        FakeEcr(),
        destination_region="us-west-2",
        repository_name="generals-training",
    )


def test_fallback_cancels_and_waits_for_terminal_state_before_resubmitting():
    class FakeBatch:
        cancelled = False

        def describe_jobs(self, *, jobs):
            assert jobs == ["job-1"]
            status = "FAILED" if self.cancelled else "RUNNABLE"
            return {"jobs": [{"jobId": "job-1", "status": status, "createdAt": 2}]}

        def list_jobs(self, **_kwargs):
            return {"jobSummaryList": []}

        def cancel_job(self, **kwargs):
            assert kwargs["jobId"] == "job-1"
            self.cancelled = True

    batch = FakeBatch()
    session = Namespace(client=lambda service: batch if service == "batch" else None)
    submission = SubmittedJob(
        experiment_id="experiment-1",
        job_id="job-1",
        job_queue_arn="queue-1",
        region="us-east-1",
    )

    assert _cancel_stalled_job_for_fallback(session, submission, 0)
    assert batch.cancelled


def test_high_memory_jobs_target_the_dedicated_h100_spot_queue():
    config = Namespace(
        high_memory_spot_job_queue_arn="h100-spot-queue",
        high_memory_spot_max_vcpus=16,
    )

    assert _job_target(config, WORKLOAD_CLASSES["high_memory_gpu"], "spot") == (
        "h100-spot-queue",
        16,
    )


def test_job_safeguards_reject_unbounded_timeout():
    args = Namespace(attempts=1, timeout_seconds=604801)

    with pytest.raises(ValueError, match="timeout-seconds"):
        _job_safeguards(args, WORKLOAD_CLASSES["high_memory_gpu"])


def test_s3_uri_builds_experiment_keys():
    uri = S3Uri.parse("s3://bucket/generals/experiments/run-1/")
    assert uri.bucket == "bucket"
    assert uri.key("/metrics/result.json") == "generals/experiments/run-1/metrics/result.json"


def test_resume_checkpoint_is_added_to_main_training_command():
    checkpoint = Path("/workspace/resume/model.eqx")
    command = _prepare_command(
        ["python", "scripts/train_ppo.py", "--config", "configs/default.yaml"],
        checkpoint,
    )
    assert command[-2:] == ["--init_checkpoint", str(checkpoint)]


def test_resume_placeholder_supports_custom_commands():
    checkpoint = Path("/workspace/resume/model.eqx")
    command = _prepare_command(["python", "custom.py", "{resume_from}"], checkpoint)
    assert command == ["python", "custom.py", str(checkpoint)]


def _matrix_args(**overrides):
    defaults = dict(
        run_prefix="exp3x3",
        config="configs/experiments/L_7d_gae90_8k.yaml",
        methods="scratch",
        seeds="44,45,46",
        num_iters=2000,
        regions="us-east-1,us-west-2",
        encoder_checkpoint=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_matrix_round_robins_seeds_across_regions():
    plan = _matrix_plan(_matrix_args())

    assert [spec["region"] for spec in plan] == ["us-east-1", "us-west-2", "us-east-1"]
    assert [spec["seed"] for spec in plan] == [44, 45, 46]
    for spec in plan:
        assert spec["resume_from"] is None
        assert "--num_iters" in spec["command"]
        assert spec["command"][spec["command"].index("--seed") + 1] == str(spec["seed"])


def test_matrix_pretrain_substitutes_per_seed_encoder():
    plan = _matrix_plan(
        _matrix_args(
            methods="scratch,pretrain",
            encoder_checkpoint="s3://bucket/encoders/belief_seed{seed}.eqx",
        )
    )

    assert len(plan) == 6
    pretrain = [spec for spec in plan if spec["method"] == "pretrain"]
    assert [spec["resume_from"] for spec in pretrain] == [
        "s3://bucket/encoders/belief_seed44.eqx",
        "s3://bucket/encoders/belief_seed45.eqx",
        "s3://bucket/encoders/belief_seed46.eqx",
    ]
    for spec in pretrain:
        assert spec["command"][-2:] == ["--init_encoder_checkpoint", "{resume_from}"]


def test_matrix_rejects_unimplemented_bc_method():
    with pytest.raises(ValueError, match="behavior cloning is not implemented"):
        _matrix_plan(_matrix_args(methods="scratch,bc"))


def test_matrix_pretrain_requires_encoder_checkpoint():
    with pytest.raises(ValueError, match="encoder-checkpoint"):
        _matrix_plan(_matrix_args(methods="pretrain"))


def test_matrix_rejects_regions_without_a_deployed_stack():
    with pytest.raises(ValueError, match="no deployed Batch stack"):
        _matrix_plan(_matrix_args(regions="us-east-2"))
