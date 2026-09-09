"""AWS Batch container wrapper that keeps experiment artifacts in S3."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import S3UploadFailedError
from botocore.exceptions import BotoCoreError, ClientError

ARTIFACT_SYNC_ERROR = 75
ARTIFACT_EXCEPTIONS = (BotoCoreError, ClientError, OSError, S3UploadFailedError, ValueError)


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    prefix: str

    @classmethod
    def parse(cls, value: str) -> S3Uri:
        if not value.startswith("s3://"):
            raise ValueError(f"Expected an s3:// URI, got {value!r}")
        bucket, separator, prefix = value[5:].partition("/")
        if not bucket:
            raise ValueError(f"S3 URI has no bucket: {value!r}")
        return cls(bucket=bucket, prefix=prefix.rstrip("/") if separator else "")

    def key(self, suffix: str) -> str:
        return "/".join(part.strip("/") for part in (self.prefix, suffix) if part.strip("/"))


class ArtifactSync:
    def __init__(self, client: Any, destination: S3Uri, root: Path) -> None:
        self.client = client
        self.destination = destination
        self.root = root
        self._uploaded: dict[Path, tuple[int, int]] = {}

    def put_json(self, suffix: str, value: dict[str, Any]) -> None:
        self.client.put_object(
            Bucket=self.destination.bucket,
            Key=self.destination.key(suffix),
            Body=(json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
            ContentType="application/json",
        )

    def sync(self, *, final: bool = False) -> int:
        uploaded = 0
        now = time.time()
        for local_name, remote_name in (("checkpoints", "checkpoints"), ("artifacts", "artifacts")):
            local_root = self.root / local_name
            if not local_root.exists():
                continue
            for path in sorted(local_root.rglob("*")):
                if not path.is_file():
                    continue
                stat = path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                if not final and now - stat.st_mtime < 30:
                    continue
                if self._uploaded.get(path) == signature:
                    continue
                relative = path.relative_to(local_root).as_posix()
                self.client.upload_file(
                    str(path),
                    self.destination.bucket,
                    self.destination.key(f"{remote_name}/{relative}"),
                )
                self._uploaded[path] = signature
                uploaded += 1
                if local_name == "checkpoints" and path.name == "config.yaml":
                    self.client.upload_file(
                        str(path),
                        self.destination.bucket,
                        self.destination.key(f"config/{path.parent.name}.yaml"),
                    )
        return uploaded


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _download_resume(client: Any, uri: str, root: Path) -> Path:
    source = S3Uri.parse(uri)
    if not source.prefix:
        raise ValueError("--resume-from must identify an S3 object, not a bucket")
    destination = root / "resume" / Path(source.prefix).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(source.bucket, source.prefix, str(destination))
    return destination


def _prepare_command(
    command: list[str],
    resume_path: Path | None,
    resume_ema_path: Path | None = None,
) -> list[str]:
    substitutions: dict[str, str] = {}
    if resume_path is not None:
        substitutions["{resume_from}"] = str(resume_path)
    if resume_ema_path is not None:
        substitutions["{resume_ema}"] = str(resume_ema_path)
    if not substitutions:
        return command
    prepared = [substitutions.get(token, token) for token in command]
    has_placeholder = prepared != command
    is_main_training = any(Path(token).name == "train_ppo.py" for token in prepared)
    if (
        resume_path is not None
        and is_main_training
        and not has_placeholder
        and "--init_checkpoint" not in prepared
    ):
        prepared.extend(["--init_checkpoint", str(resume_path)])
    return prepared


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main() -> int:
    args = _parse_args()
    destination_value = os.environ.get("EXPERIMENT_S3_PREFIX")
    if not destination_value:
        print("EXPERIMENT_S3_PREFIX is required", file=sys.stderr)
        return ARTIFACT_SYNC_ERROR

    experiment_id = os.environ.get("EXPERIMENT_ID", "unknown")
    root = Path(os.environ.get("EXPERIMENT_WORKDIR", "/workspace"))
    interval = max(10, int(os.environ.get("ARTIFACT_SYNC_INTERVAL_SECONDS", "300")))
    s3 = boto3.client("s3")
    sync = ArtifactSync(s3, S3Uri.parse(destination_value), root)

    resume_path = None
    resume_ema_path = None
    resume_from = os.environ.get("RESUME_FROM")
    resume_ema_from = os.environ.get("RESUME_EMA_FROM")
    try:
        if resume_from:
            resume_path = _download_resume(s3, resume_from, root)
            print(f"Downloaded resume checkpoint to {resume_path}", flush=True)
        if resume_ema_from:
            resume_ema_path = _download_resume(s3, resume_ema_from, root)
            print(f"Downloaded EMA resume checkpoint to {resume_ema_path}", flush=True)
        command = _prepare_command(args.command, resume_path, resume_ema_path)
        sync.put_json(
            "config/job.json",
            {
                "aws_batch_job_id": os.environ.get("AWS_BATCH_JOB_ID"),
                "aws_batch_job_attempt": os.environ.get("AWS_BATCH_JOB_ATTEMPT"),
                "command": command,
                "experiment_id": experiment_id,
                "image": os.environ.get("CONTAINER_IMAGE"),
                "resume_from": resume_from,
                "resume_ema_from": resume_ema_from,
                "started_at": _utc_now(),
            },
        )
    except ARTIFACT_EXCEPTIONS as error:
        print(f"Unable to initialize S3 experiment state: {error}", file=sys.stderr, flush=True)
        return ARTIFACT_SYNC_ERROR

    child_env = os.environ.copy()
    if resume_path is not None:
        child_env["RESUME_FROM_LOCAL"] = str(resume_path)
    if resume_ema_path is not None:
        child_env["RESUME_EMA_FROM_LOCAL"] = str(resume_ema_path)

    print(f"Experiment: {experiment_id}", flush=True)
    print(f"Artifacts: {destination_value}", flush=True)
    print(f"Command: {command}", flush=True)

    process = subprocess.Popen(command, cwd=root, env=child_env)
    interrupted_by: int | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal interrupted_by
        interrupted_by = signum
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    next_sync = time.monotonic() + interval
    while process.poll() is None:
        time.sleep(1)
        if time.monotonic() >= next_sync:
            try:
                count = sync.sync()
                if count:
                    print(f"Uploaded {count} stable artifact(s)", flush=True)
            except ARTIFACT_EXCEPTIONS as error:
                print(f"Periodic S3 artifact sync failed: {error}", file=sys.stderr, flush=True)
            next_sync = time.monotonic() + interval

    exit_code = int(process.returncode or 0)
    finished_at = _utc_now()
    try:
        uploaded = sync.sync(final=True)
        sync.put_json(
            "metrics/result.json",
            {
                "command": command,
                "exit_code": exit_code,
                "experiment_id": experiment_id,
                "finished_at": finished_at,
                "interrupted_by_signal": interrupted_by,
                "uploaded_artifacts": uploaded,
            },
        )
    except ARTIFACT_EXCEPTIONS as error:
        print(f"Final S3 artifact sync failed: {error}", file=sys.stderr, flush=True)
        return ARTIFACT_SYNC_ERROR

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
