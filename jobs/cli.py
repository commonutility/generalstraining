"""Small AWS Batch interface for Generals training jobs."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

TERMINAL_STATES = {"SUCCEEDED", "FAILED"}
DEFAULT_STACK = "GeneralsTrainingBatch"
REGIONAL_STACKS = {
    "us-east-1": DEFAULT_STACK,
    "us-west-2": "GeneralsTrainingBatchUsWest2",
}
MATRIX_METHODS = {
    # method name -> whether it requires a pretrained encoder checkpoint
    "scratch": False,
    "pretrain": True,
}
MATRIX_DEFAULT_TIMEOUT_SECONDS = 259200  # 72 hours; ~2k iterations at 100.5s each is ~56h
RESUME_OVERLAY_BOOTSTRAP = """\
import os
import pathlib
import sys
import zipfile

import boto3


def download(uri, destination):
    bucket, key = uri[5:].split("/", 1)
    boto3.client("s3").download_file(bucket, key, str(destination))


full_checkpoint, ema_uri, overlay_uri, train_script, *train_args = sys.argv[1:]
root = pathlib.Path("/tmp/generals-resume-overlay")
root.mkdir(parents=True, exist_ok=True)
archive = root / "source.zip"
ema_checkpoint = root / pathlib.Path(ema_uri).name
download(overlay_uri, archive)
download(ema_uri, ema_checkpoint)
with zipfile.ZipFile(archive) as source:
    source.extractall(root)
os.environ["PYTHONPATH"] = str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
os.execv(
    sys.executable,
    [
        sys.executable,
        train_script,
        *train_args,
        "--init_checkpoint",
        full_checkpoint,
        "--ema_checkpoint",
        str(ema_checkpoint),
    ],
)
"""
WORKLOAD_CLASSES = {
    "small_gpu": {
        "cpus": 4,
        "memory": 15000,
        "max_memory": 15000,
        "gpu": 1,
        "queue": "standard",
        "default_attempts": None,
        "default_timeout_seconds": None,
        "compatible_instances": ("g6e.xlarge", "g6.xlarge", "g5.xlarge"),
    },
    "high_memory_gpu": {
        "cpus": 4,
        "memory": 56000,
        "max_memory": 60000,
        "gpu": 1,
        "queue": "high_memory",
        "default_attempts": 1,
        "default_timeout_seconds": 7200,
        "compatible_instances": ("g7e.2xlarge", "p5.4xlarge", "p4de.24xlarge"),
    },
}
SENSITIVE_ENV_NAMES = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
}


@dataclass(frozen=True)
class StackConfig:
    artifact_bucket: str
    image_tag: str
    high_memory_job_queue_arn: str
    high_memory_max_vcpus: int
    high_memory_spot_job_queue_arn: str
    high_memory_spot_max_vcpus: int
    job_definition_arn: str
    job_queue_arn: str
    max_vcpus: int
    repository_uri: str
    spot_job_queue_arn: str
    spot_max_vcpus: int

    @property
    def image(self) -> str:
        return f"{self.repository_uri}:{self.image_tag}"


@dataclass(frozen=True)
class SubmittedJob:
    experiment_id: str
    job_id: str
    job_queue_arn: str
    region: str


def _sanitize_name(value: str, *, max_length: int = 128) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    if not sanitized:
        raise ValueError("name must contain at least one letter or number")
    return sanitized[:max_length]


def _experiment_id(name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{_sanitize_name(name, max_length=80)}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _parse_environment(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(f"environment value must be KEY=VALUE: {value!r}")
        if key.startswith("AWS_BATCH"):
            raise ValueError(f"AWS Batch reserves environment name {key!r}")
        if key in SENSITIVE_ENV_NAMES:
            raise ValueError(f"do not pass temporary AWS credentials through --env ({key})")
        result[key] = item
    return result


def _session(args: argparse.Namespace) -> boto3.Session:
    return boto3.Session(profile_name=args.profile, region_name=args.region)


def _region(session: boto3.Session) -> str:
    return session.region_name or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"


def _regional_stack_name(region: str, explicit_stack: str | None = None) -> str:
    if explicit_stack:
        return explicit_stack
    return REGIONAL_STACKS.get(region, DEFAULT_STACK)


def _load_stack_config(session: boto3.Session, stack_name: str) -> StackConfig:
    response = session.client("cloudformation").describe_stacks(StackName=stack_name)
    outputs = {
        item["OutputKey"]: item["OutputValue"]
        for item in response["Stacks"][0].get("Outputs", [])
    }
    required = {
        "ArtifactBucketName",
        "HighMemoryJobQueueArn",
        "HighMemoryMaxVCpus",
        "HighMemorySpotJobQueueArn",
        "HighMemorySpotMaxVCpus",
        "ImageTag",
        "JobDefinitionArn",
        "JobQueueArn",
        "MaxVCpus",
        "RepositoryUri",
        "SpotJobQueueArn",
        "SpotMaxVCpus",
    }
    missing = sorted(required - outputs.keys())
    if missing:
        raise RuntimeError(f"{stack_name} is missing outputs: {', '.join(missing)}")
    return StackConfig(
        artifact_bucket=outputs["ArtifactBucketName"],
        high_memory_job_queue_arn=outputs["HighMemoryJobQueueArn"],
        high_memory_max_vcpus=int(outputs["HighMemoryMaxVCpus"]),
        high_memory_spot_job_queue_arn=outputs["HighMemorySpotJobQueueArn"],
        high_memory_spot_max_vcpus=int(outputs["HighMemorySpotMaxVCpus"]),
        image_tag=outputs["ImageTag"],
        job_definition_arn=outputs["JobDefinitionArn"],
        job_queue_arn=outputs["JobQueueArn"],
        max_vcpus=int(outputs["MaxVCpus"]),
        repository_uri=outputs["RepositoryUri"],
        spot_job_queue_arn=outputs["SpotJobQueueArn"],
        spot_max_vcpus=int(outputs["SpotMaxVCpus"]),
    )


def _describe_job(batch: Any, job_id: str) -> dict[str, Any]:
    jobs = batch.describe_jobs(jobs=[job_id]).get("jobs", [])
    if not jobs:
        raise RuntimeError(f"AWS Batch job not found: {job_id}")
    return jobs[0]


def _command_tokens(command: str) -> list[str]:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("--command cannot be empty")
    return tokens


def _job_safeguards(args: argparse.Namespace, workload: dict[str, Any]) -> dict[str, Any]:
    attempts = args.attempts if args.attempts is not None else workload["default_attempts"]
    timeout_seconds = (
        args.timeout_seconds
        if args.timeout_seconds is not None
        else workload["default_timeout_seconds"]
    )
    if attempts is not None and not 1 <= attempts <= 10:
        raise ValueError("--attempts must be between 1 and 10")
    if timeout_seconds is not None and not 60 <= timeout_seconds <= 604800:
        raise ValueError("--timeout-seconds must be between 60 and 604800")

    safeguards: dict[str, Any] = {}
    if attempts is not None:
        safeguards["retryStrategy"] = {"attempts": attempts}
    if timeout_seconds is not None:
        safeguards["timeout"] = {"attemptDurationSeconds": timeout_seconds}
    return safeguards


def _job_target(
    config: StackConfig,
    workload: dict[str, Any],
    compute: str,
) -> tuple[str, int]:
    if compute == "spot":
        if workload["queue"] == "high_memory":
            return config.high_memory_spot_job_queue_arn, config.high_memory_spot_max_vcpus
        return config.spot_job_queue_arn, config.spot_max_vcpus
    if compute != "ondemand":
        raise ValueError(f"unsupported compute mode: {compute}")
    if workload["queue"] == "high_memory":
        return config.high_memory_job_queue_arn, config.high_memory_max_vcpus
    return config.job_queue_arn, config.max_vcpus


def _submit_once(
    args: argparse.Namespace,
    session: boto3.Session,
    stack_name: str,
    experiment_id: str,
    command: list[str],
    *,
    fallback_from_job_id: str | None = None,
) -> SubmittedJob:
    region = _region(session)
    config = _load_stack_config(session, stack_name)
    workload = WORKLOAD_CLASSES[args.workload_class]
    cpus = args.cpus if args.cpus is not None else workload["cpus"]
    memory = args.memory if args.memory is not None else workload["memory"]
    gpu = args.gpu if args.gpu is not None else workload["gpu"]
    job_queue_arn, max_vcpus = _job_target(config, workload, args.compute)

    if cpus < 1 or cpus > max_vcpus:
        raise ValueError(
            f"requested {cpus} vCPUs, but the deployed environment supports at most "
            f"{max_vcpus} for {args.workload_class}"
        )
    if memory < 4 or memory > workload["max_memory"]:
        raise ValueError(
            f"the deployed {args.workload_class} class supports 4-{workload['max_memory']} MiB"
        )
    if gpu != 1:
        raise ValueError("the deployed GPU job definition currently supports exactly one GPU")
    safeguards = _job_safeguards(args, workload)

    s3_prefix = f"s3://{config.artifact_bucket}/generals/experiments/{experiment_id}"
    environment = _parse_environment(args.env)
    environment.update(
        {
            "CONTAINER_IMAGE": config.image,
            "EXPERIMENT_ID": experiment_id,
            "EXPERIMENT_S3_PREFIX": s3_prefix,
        }
    )
    if args.resume_from:
        environment["RESUME_FROM"] = args.resume_from
    if getattr(args, "resume_ema_from", None):
        environment["RESUME_EMA_FROM"] = args.resume_ema_from
    if fallback_from_job_id:
        environment["FALLBACK_FROM_JOB_ID"] = fallback_from_job_id

    batch = session.client("batch")
    tags = {
        "ExperimentId": experiment_id,
        "Project": "GeneralsTraining",
        "Compute": args.compute,
        "Region": region,
        "WorkloadClass": args.workload_class,
    }
    if fallback_from_job_id:
        tags["FallbackFrom"] = fallback_from_job_id
    response = batch.submit_job(
        jobName=_sanitize_name(args.name),
        jobQueue=job_queue_arn,
        jobDefinition=config.job_definition_arn,
        containerOverrides={
            "command": ["python", "jobs/runtime.py", "--", *command],
            "environment": [{"name": key, "value": value} for key, value in sorted(environment.items())],
            "resourceRequirements": [
                {"type": "VCPU", "value": str(cpus)},
                {"type": "MEMORY", "value": str(memory)},
                {"type": "GPU", "value": str(gpu)},
            ],
        },
        tags=tags,
        **safeguards,
    )

    print(f"Experiment ID: {experiment_id}")
    print(f"AWS Batch Job ID: {response['jobId']}")
    print(f"AWS region: {region}")
    print(f"Container image: {config.image}")
    print(f"Compute: {args.compute}")
    print(f"Workload class: {args.workload_class}")
    if "retryStrategy" in safeguards:
        print(f"Maximum attempts: {safeguards['retryStrategy']['attempts']}")
    if "timeout" in safeguards:
        print(f"Attempt timeout: {safeguards['timeout']['attemptDurationSeconds']} seconds")
    print(f"Submitted command: {shlex.join(command)}")
    print(f"S3 output prefix: {s3_prefix}")
    return SubmittedJob(
        experiment_id=experiment_id,
        job_id=response["jobId"],
        job_queue_arn=job_queue_arn,
        region=region,
    )


def _cancel_stalled_job_for_fallback(
    session: boto3.Session,
    submission: SubmittedJob,
    wait_seconds: int,
) -> bool:
    batch = session.client("batch")
    deadline = time.monotonic() + wait_seconds
    job: dict[str, Any]
    while True:
        job = _describe_job(batch, submission.job_id)
        if job["status"] in TERMINAL_STATES | {"STARTING", "RUNNING"}:
            print(
                f"Preferred-region job reached {job['status']}; "
                "regional fallback was not submitted"
            )
            return False
        if time.monotonic() >= deadline:
            break
        time.sleep(15)

    older_jobs: list[dict[str, Any]] = []
    for status_name in ("SUBMITTED", "PENDING", "RUNNABLE"):
        next_token = None
        while True:
            request = {
                "jobQueue": submission.job_queue_arn,
                "jobStatus": status_name,
            }
            if next_token:
                request["nextToken"] = next_token
            response = batch.list_jobs(**request)
            older_jobs.extend(
                summary
                for summary in response.get("jobSummaryList", [])
                if summary["jobId"] != submission.job_id
                and summary.get("createdAt", 0) < job.get("createdAt", 0)
            )
            next_token = response.get("nextToken")
            if not next_token:
                break
    if older_jobs:
        print(
            "Preferred queue has older active jobs; keeping the submitted job in place "
            "instead of treating queue delay as a regional capacity failure"
        )
        return False

    batch.cancel_job(
        jobId=submission.job_id,
        reason="Cancelled after bounded RUNNABLE wait for regional fallback",
    )
    cancel_deadline = time.monotonic() + 180
    while time.monotonic() < cancel_deadline:
        job = _describe_job(batch, submission.job_id)
        if job["status"] == "FAILED":
            print(f"Cancelled stalled preferred-region job {submission.job_id}")
            return True
        if job["status"] in {"STARTING", "RUNNING", "SUCCEEDED"}:
            print(
                f"Preferred-region job reached {job['status']} during cancellation; "
                "regional fallback was not submitted"
            )
            return False
        time.sleep(5)
    raise RuntimeError(
        "preferred-region job did not reach a terminal state after cancellation; "
        "refusing to create a duplicate fallback job"
    )


def submit(args: argparse.Namespace) -> int:
    session = _session(args)
    region = _region(session)
    if not args.fallback_region and args.fallback_after_seconds is not None:
        raise ValueError("--fallback-after-seconds requires --fallback-region")
    if args.fallback_region == region:
        raise ValueError("--fallback-region must differ from the preferred AWS region")
    fallback_after = args.fallback_after_seconds or 900
    if args.fallback_region and not 300 <= fallback_after <= 86400:
        raise ValueError("--fallback-after-seconds must be between 300 and 86400")

    stack_name = _regional_stack_name(region, args.stack)
    command = _command_tokens(args.command)
    experiment_id = _sanitize_name(args.experiment_id or _experiment_id(args.name))
    submission = _submit_once(args, session, stack_name, experiment_id, command)

    if not args.fallback_region:
        return 0

    print(
        f"Waiting up to {fallback_after} seconds for the preferred-region job to start "
        f"before considering {args.fallback_region}"
    )
    if not _cancel_stalled_job_for_fallback(session, submission, fallback_after):
        return 0

    fallback_session = boto3.Session(
        profile_name=args.profile,
        region_name=args.fallback_region,
    )
    fallback_stack = _regional_stack_name(args.fallback_region, args.fallback_stack)
    _submit_once(
        args,
        fallback_session,
        fallback_stack,
        experiment_id,
        command,
        fallback_from_job_id=submission.job_id,
    )
    return 0


def _ema_checkpoint_uri(uri: str) -> str:
    """Derive the EMA checkpoint URI saved beside a full milestone checkpoint.

    The trainer writes `<run>_<iter>.eqx` (network + optimizer state) and
    `<run>_ema_<iter>.eqx` side by side at each save_at milestone.
    """
    match = re.fullmatch(r"(.+)_(\d+)\.eqx", uri)
    if not match:
        raise ValueError(
            f"cannot derive the EMA checkpoint URI from {uri!r}; "
            "expected an s3://.../<run>_<iteration>.eqx milestone checkpoint"
        )
    return f"{match.group(1)}_ema_{match.group(2)}.eqx"


def _parse_resume_checkpoints(values: list[str] | None) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values or []:
        seed_str, separator, uri = value.partition("=")
        if not separator or not seed_str.strip().isdigit() or not uri.startswith("s3://"):
            raise ValueError(
                f"--resume-checkpoints entries must be SEED=s3://... : {value!r}"
            )
        result[int(seed_str)] = uri
    return result


def _matrix_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Expand {method x seed} into concrete job specs, round-robined over regions."""
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("--methods cannot be empty")
    for method in methods:
        if method == "bc":
            raise ValueError(
                "behavior cloning is not implemented in this repository yet "
                "(docs/EXPERIMENTS.md); submit BC jobs once a BC trainer exists"
            )
        if method not in MATRIX_METHODS:
            raise ValueError(f"unknown method {method!r}; supported: {sorted(MATRIX_METHODS)}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    if not regions:
        raise ValueError("--regions cannot be empty")
    for region in regions:
        if region not in REGIONAL_STACKS:
            raise ValueError(
                f"no deployed Batch stack is known for {region}; "
                f"supported regions: {sorted(REGIONAL_STACKS)}"
            )
    if any(MATRIX_METHODS[m] for m in methods) and not args.encoder_checkpoint:
        raise ValueError(
            "--encoder-checkpoint (S3 URI, may contain {seed}) is required for the "
            "pretrain method"
        )

    resume_map = _parse_resume_checkpoints(getattr(args, "resume_checkpoints", None))
    iteration_offset = getattr(args, "iteration_offset", None)
    source_overlay = getattr(args, "source_overlay", None)
    if resume_map and iteration_offset is None:
        raise ValueError(
            "--resume-checkpoints requires --iteration-offset (the global iteration "
            "count already completed by the checkpoints)"
        )
    if iteration_offset is not None and not resume_map:
        raise ValueError("--iteration-offset requires --resume-checkpoints")
    if source_overlay and not resume_map:
        raise ValueError("--source-overlay requires --resume-checkpoints")
    if resume_map:
        if any(MATRIX_METHODS[m] for m in methods):
            raise ValueError(
                "--resume-checkpoints only supports the scratch method; "
                "init_checkpoint and init_encoder_checkpoint are mutually exclusive"
            )
        missing = [seed for seed in seeds if seed not in resume_map]
        if missing:
            raise ValueError(f"--resume-checkpoints is missing entries for seeds {missing}")

    plan: list[dict[str, Any]] = []
    index = 0
    for method in methods:
        for seed in seeds:
            run_name = _sanitize_name(f"{args.run_prefix}_{method}_seed{seed}").replace("-", "_")
            command = [
                "python", "scripts/train_ppo.py",
                "--config", args.config,
                "--seed", str(seed),
                "--run_name", run_name,
            ]
            if args.num_iters is not None:
                command += ["--num_iters", str(args.num_iters)]
            if getattr(args, "save_at", None):
                command += ["--save_at", *(str(i) for i in args.save_at)]
            resume_from = None
            resume_ema_from = None
            if MATRIX_METHODS[method]:
                # jobs/runtime.py downloads RESUME_FROM and substitutes the local
                # path for the {resume_from} token before launching training.
                resume_from = args.encoder_checkpoint.replace("{seed}", str(seed))
                command += ["--init_encoder_checkpoint", "{resume_from}"]
            elif resume_map:
                # Resume weights + optimizer state and the matching EMA weights;
                # iteration_offset keeps schedules, logging, and save_at global.
                resume_from = resume_map[seed]
                ema_uri = _ema_checkpoint_uri(resume_from)
                if source_overlay:
                    # Compatibility path for a deployed image that predates the
                    # resume-bookkeeping changes. The old runtime downloads the
                    # full checkpoint; this bootstrap downloads the EMA and a
                    # small source overlay before execing the trainer.
                    command = [
                        "python",
                        "-c",
                        RESUME_OVERLAY_BOOTSTRAP,
                        "{resume_from}",
                        ema_uri,
                        source_overlay,
                        "scripts/train_ppo.py",
                        *command[2:],
                        "--iteration_offset",
                        str(iteration_offset),
                    ]
                else:
                    resume_ema_from = ema_uri
                    command += [
                        "--init_checkpoint", "{resume_from}",
                        "--ema_checkpoint", "{resume_ema}",
                        "--iteration_offset", str(iteration_offset),
                    ]
            plan.append(
                {
                    "method": method,
                    "seed": seed,
                    "region": regions[index % len(regions)],
                    "run_name": run_name,
                    "command": command,
                    "resume_from": resume_from,
                    "resume_ema_from": resume_ema_from,
                }
            )
            index += 1
    return plan


def submit_matrix(args: argparse.Namespace) -> int:
    plan = _matrix_plan(args)

    if args.dry_run:
        print(f"Planned {len(plan)} jobs (dry run, nothing submitted):")
        for spec in plan:
            print(
                f"  {spec['method']:>8} seed={spec['seed']} region={spec['region']} "
                f"run={spec['run_name']}"
            )
            print(f"           command: {shlex.join(spec['command'])}")
            if spec["resume_from"]:
                label = "resume" if spec["resume_ema_from"] else "encoder"
                print(f"           {label}: {spec['resume_from']}")
            if spec["resume_ema_from"]:
                print(f"           resume EMA: {spec['resume_ema_from']}")
        return 0

    sessions: dict[str, boto3.Session] = {}
    submitted: list[tuple[dict[str, Any], SubmittedJob]] = []
    for spec in plan:
        region = spec["region"]
        if region not in sessions:
            sessions[region] = boto3.Session(profile_name=args.profile, region_name=region)
        job_args = argparse.Namespace(
            name=spec["run_name"].replace("_", "-"),
            workload_class=args.workload_class,
            compute=args.compute,
            gpu=args.gpu,
            cpus=args.cpus,
            memory=args.memory,
            attempts=args.attempts,
            timeout_seconds=args.timeout_seconds,
            env=list(args.env),
            resume_from=spec["resume_from"],
            resume_ema_from=spec["resume_ema_from"],
        )
        experiment_id = _sanitize_name(_experiment_id(spec["run_name"]))
        submission = _submit_once(
            job_args,
            sessions[region],
            _regional_stack_name(region),
            experiment_id,
            spec["command"],
        )
        submitted.append((spec, submission))
        print()

    print(f"Submitted {len(submitted)} jobs:")
    for spec, submission in submitted:
        print(
            f"  {spec['method']:>8} seed={spec['seed']} region={submission.region} "
            f"job={submission.job_id}"
        )
    return 0


def status(args: argparse.Namespace) -> int:
    session = _session(args)
    job = _describe_job(session.client("batch"), args.job_id)
    container = job.get("container", {})
    print(job["status"])
    print(f"Job ID: {job['jobId']}")
    print(f"Job name: {job['jobName']}")
    if job.get("statusReason"):
        print(f"Status reason: {job['statusReason']}")
    if container.get("reason"):
        print(f"Container reason: {container['reason']}")
    if "exitCode" in container:
        print(f"Exit code: {container['exitCode']}")
    if container.get("image"):
        print(f"Container image: {container['image']}")
    if container.get("logStreamName"):
        print(f"Log stream: {container['logStreamName']}")
    for index, attempt in enumerate(job.get("attempts", []), start=1):
        attempt_container = attempt.get("container", {})
        details = [
            value
            for value in (
                attempt.get("statusReason"),
                attempt_container.get("reason"),
                f"exit={attempt_container['exitCode']}" if "exitCode" in attempt_container else None,
            )
            if value
        ]
        if details:
            print(f"Attempt {index}: {'; '.join(details)}")
    return 0


def _wait_for_log_stream(batch: Any, job_id: str, follow: bool) -> dict[str, Any]:
    while True:
        job = _describe_job(batch, job_id)
        if job.get("container", {}).get("logStreamName"):
            return job
        if not follow or job["status"] in TERMINAL_STATES:
            return job
        print(f"Waiting for log stream ({job['status']})...", file=sys.stderr)
        time.sleep(5)


def logs(args: argparse.Namespace) -> int:
    session = _session(args)
    batch = session.client("batch")
    job = _wait_for_log_stream(batch, args.job_id, args.follow)
    container = job.get("container", {})
    stream = container.get("logStreamName")
    if not stream:
        print(f"No log stream is available yet; job status is {job['status']}", file=sys.stderr)
        if job.get("statusReason"):
            print(job["statusReason"], file=sys.stderr)
        return 0

    log_options = container.get("logConfiguration", {}).get("options", {})
    log_group = log_options.get("awslogs-group", "/aws/batch/generals-training")
    client = session.client("logs")
    token = None
    terminal_unchanged = 0
    while True:
        request: dict[str, Any] = {
            "logGroupName": log_group,
            "logStreamName": stream,
            "startFromHead": True,
        }
        if token:
            request["nextToken"] = token
        response = client.get_log_events(**request)
        for event in response.get("events", []):
            print(event["message"])
        next_token = response.get("nextForwardToken")
        unchanged = next_token == token
        token = next_token
        if not args.follow:
            break
        job = _describe_job(batch, args.job_id)
        if job["status"] in TERMINAL_STATES and unchanged:
            terminal_unchanged += 1
            if terminal_unchanged >= 2:
                break
        else:
            terminal_unchanged = 0
        time.sleep(3)
    return 0


def cancel(args: argparse.Namespace) -> int:
    session = _session(args)
    batch = session.client("batch")
    job = _describe_job(batch, args.job_id)
    reason = args.reason or "Cancelled through jobs/cli.py"
    if job["status"] in TERMINAL_STATES:
        print(f"Job is already {job['status']}; no action taken")
        return 0
    if job["status"] in {"STARTING", "RUNNING"}:
        batch.terminate_job(jobId=args.job_id, reason=reason)
        print(f"Termination requested for {args.job_id}")
    else:
        batch.cancel_job(jobId=args.job_id, reason=reason)
        print(f"Cancellation requested for {args.job_id}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="AWS CLI profile; defaults to the normal AWS credential chain")
    parser.add_argument("--region", help="AWS region; defaults to the active profile region")
    parser.add_argument(
        "--stack",
        help="Batch CloudFormation stack; defaults to the deployed stack for --region",
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit a GPU training job")
    submit_parser.add_argument("--name", required=True)
    submit_parser.add_argument("--command", required=True)
    submit_parser.add_argument("--experiment-id")
    submit_parser.add_argument("--workload-class", choices=sorted(WORKLOAD_CLASSES), default="small_gpu")
    submit_parser.add_argument("--compute", choices=("ondemand", "spot"), default="ondemand")
    submit_parser.add_argument("--gpu", type=int)
    submit_parser.add_argument("--cpus", type=int)
    submit_parser.add_argument("--memory", type=int, help="Container memory in MiB")
    submit_parser.add_argument("--attempts", type=int, help="Maximum Batch attempts (high-memory default: 1)")
    submit_parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Hard timeout for each attempt (high-memory default: 7200)",
    )
    submit_parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    submit_parser.add_argument("--resume-from", help="S3 object URI for a checkpoint")
    submit_parser.add_argument(
        "--resume-ema-from",
        help="S3 object URI for an EMA checkpoint (substituted for {resume_ema})",
    )
    submit_parser.add_argument(
        "--fallback-region",
        help="Cancel a capacity-stalled preferred job and resubmit once in this region",
    )
    submit_parser.add_argument(
        "--fallback-stack",
        help="Fallback CloudFormation stack; defaults to the deployed stack for --fallback-region",
    )
    submit_parser.add_argument(
        "--fallback-after-seconds",
        type=int,
        help="RUNNABLE wait before safe fallback (default: 900; minimum: 300)",
    )
    submit_parser.set_defaults(handler=submit)

    matrix_parser = subparsers.add_parser(
        "submit-matrix",
        help="Submit a {method x seed} experiment matrix, one GPU per job, across regions",
    )
    matrix_parser.add_argument("--run-prefix", required=True, help="Prefix for run names")
    matrix_parser.add_argument("--config", required=True, help="Training YAML, e.g. configs/experiments/L_7d_gae90_8k.yaml")
    matrix_parser.add_argument(
        "--methods",
        default="scratch",
        help=f"Comma-separated methods ({', '.join(sorted(MATRIX_METHODS))})",
    )
    matrix_parser.add_argument("--seeds", required=True, help="Comma-separated integer seeds, e.g. 44,45,46")
    matrix_parser.add_argument("--num-iters", type=int, help="Override num_iters for every job")
    matrix_parser.add_argument(
        "--save-at",
        type=int,
        nargs="+",
        help="Override save_at checkpoint iterations, e.g. --save-at 1000 2000",
    )
    matrix_parser.add_argument(
        "--regions",
        default="us-east-1",
        help=f"Comma-separated regions to round-robin over ({', '.join(sorted(REGIONAL_STACKS))})",
    )
    matrix_parser.add_argument(
        "--encoder-checkpoint",
        help="S3 URI of the pretrained encoder for the pretrain method; {seed} is substituted",
    )
    matrix_parser.add_argument(
        "--resume-checkpoints",
        nargs="+",
        metavar="SEED=S3URI",
        help="Per-seed full milestone checkpoints (network + optimizer state) to resume "
        "from; the matching _ema_ checkpoint URI is derived automatically",
    )
    matrix_parser.add_argument(
        "--iteration-offset",
        type=int,
        help="Global iterations already completed by --resume-checkpoints; keeps "
        "schedules, logged iterations, and --save-at milestones global",
    )
    matrix_parser.add_argument(
        "--source-overlay",
        help="S3 URI of a source zip to apply before a resumed run; compatibility "
        "path for a deployed image that predates resume bookkeeping",
    )
    matrix_parser.add_argument("--workload-class", choices=sorted(WORKLOAD_CLASSES), default="high_memory_gpu")
    matrix_parser.add_argument("--compute", choices=("ondemand", "spot"), default="ondemand")
    matrix_parser.add_argument("--gpu", type=int)
    matrix_parser.add_argument("--cpus", type=int)
    matrix_parser.add_argument("--memory", type=int, help="Container memory in MiB")
    matrix_parser.add_argument("--attempts", type=int)
    matrix_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=MATRIX_DEFAULT_TIMEOUT_SECONDS,
        help=f"Hard timeout per attempt (default: {MATRIX_DEFAULT_TIMEOUT_SECONDS})",
    )
    matrix_parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    matrix_parser.add_argument("--dry-run", action="store_true", help="Print the matrix without submitting")
    matrix_parser.set_defaults(handler=submit_matrix)

    status_parser = subparsers.add_parser("status", help="Describe a Batch job")
    status_parser.add_argument("job_id")
    status_parser.set_defaults(handler=status)

    logs_parser = subparsers.add_parser("logs", help="Print CloudWatch logs for a Batch job")
    logs_parser.add_argument("job_id")
    logs_parser.add_argument("--follow", action="store_true")
    logs_parser.set_defaults(handler=logs)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel or terminate a Batch job")
    cancel_parser.add_argument("job_id")
    cancel_parser.add_argument("--reason")
    cancel_parser.set_defaults(handler=cancel)
    return parser


def main() -> int:
    try:
        args = _parser().parse_args()
        return int(args.handler(args))
    except (BotoCoreError, ClientError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
