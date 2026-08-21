#!/usr/bin/env bash
set -euo pipefail

PROFILE="${AWS_PROFILE:-default}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(aws configure export-credentials --profile "${PROFILE}" --format env)"
export CDK_DEFAULT_ACCOUNT="${CDK_DEFAULT_ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}"
export CDK_DEFAULT_REGION="${CDK_DEFAULT_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region)}}}"

cd "${SCRIPT_DIR}"
exec npx cdk "$@"
