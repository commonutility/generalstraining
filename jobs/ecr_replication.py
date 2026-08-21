"""Idempotently add the Generals ECR cross-region replication rule."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

MAX_REPLICATION_RULES = 10


def _rule_covers_repository(
    rule: dict[str, Any],
    *,
    destination_region: str,
    registry_id: str,
    repository_name: str,
) -> bool:
    has_destination = any(
        destination.get("region") == destination_region
        and destination.get("registryId") == registry_id
        for destination in rule.get("destinations", [])
    )
    if not has_destination:
        return False

    filters = rule.get("repositoryFilters", [])
    if not filters:
        return True
    return any(
        item.get("filterType") == "PREFIX_MATCH"
        and repository_name.startswith(item.get("filter", ""))
        for item in filters
    )


def ensure_replication_rule(
    ecr: Any,
    *,
    destination_region: str,
    repository_name: str,
) -> bool:
    """Add a filtered same-account rule while preserving unrelated rules."""
    registry = ecr.describe_registry()
    registry_id = registry["registryId"]
    rules = list(registry.get("replicationConfiguration", {}).get("rules", []))

    if any(
        _rule_covers_repository(
            rule,
            destination_region=destination_region,
            registry_id=registry_id,
            repository_name=repository_name,
        )
        for rule in rules
    ):
        return False
    if len(rules) >= MAX_REPLICATION_RULES:
        raise RuntimeError(
            f"ECR already has {len(rules)} replication rules; "
            "cannot safely add the Generals rule"
        )

    rules.append(
        {
            "destinations": [
                {
                    "region": destination_region,
                    "registryId": registry_id,
                }
            ],
            "repositoryFilters": [
                {
                    "filter": repository_name,
                    "filterType": "PREFIX_MATCH",
                }
            ],
        }
    )
    ecr.put_replication_configuration(
        replicationConfiguration={"rules": rules},
    )
    return True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-region", required=True)
    parser.add_argument("--destination-region", required=True)
    parser.add_argument("--repository-name", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        session = boto3.Session(region_name=args.source_region)
        changed = ensure_replication_rule(
            session.client("ecr"),
            destination_region=args.destination_region,
            repository_name=args.repository_name,
        )
    except (BotoCoreError, ClientError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    action = "Configured" if changed else "Found existing"
    print(
        f"{action} ECR replication for {args.repository_name}: "
        f"{args.source_region} -> {args.destination_region}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
