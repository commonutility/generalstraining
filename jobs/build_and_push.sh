#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK_NAME="${GENERALS_STORAGE_STACK:-GeneralsTrainingStorage}"
REGION="${GENERALS_PRIMARY_REGION:-us-east-1}"
FALLBACK_REGION="${GENERALS_FALLBACK_REGION:-us-west-2}"
FALLBACK_FOUNDATION_STACK="${GENERALS_FALLBACK_FOUNDATION_STACK:-GeneralsTrainingWest2Foundation}"
REPLICATION_WAIT_SECONDS="${GENERALS_ECR_REPLICATION_WAIT_SECONDS:-600}"

if [[ ! "${REPLICATION_WAIT_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "GENERALS_ECR_REPLICATION_WAIT_SECONDS must be a non-negative integer." >&2
  exit 1
fi

image_exists() {
  local region="$1"
  local repository_name="$2"
  local image_tag="$3"
  local output

  if output="$(
    aws ecr describe-images \
      --region "${region}" \
      --repository-name "${repository_name}" \
      --image-ids "imageTag=${image_tag}" \
      --query 'imageDetails[0].imageDigest' \
      --output text \
      2>&1
  )"; then
    return 0
  fi
  if [[ "${output}" == *"ImageNotFoundException"* ]]; then
    return 1
  fi
  echo "${output}" >&2
  return 2
}

REPOSITORY_URI="$(
  aws cloudformation describe-stacks \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='RepositoryUri'].OutputValue | [0]" \
    --output text
)"
if [[ -z "${REPOSITORY_URI}" || "${REPOSITORY_URI}" == "None" ]]; then
  echo "No RepositoryUri output exists on ${STACK_NAME}." >&2
  echo "Deploy the updated storage stack before building the image." >&2
  exit 1
fi
ARTIFACT_BUCKET="$(
  aws cloudformation describe-stacks \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='CheckpointBucketName'].OutputValue | [0]" \
    --output text
)"
if [[ -z "${ARTIFACT_BUCKET}" || "${ARTIFACT_BUCKET}" == "None" ]]; then
  echo "No CheckpointBucketName output exists on ${STACK_NAME}." >&2
  exit 1
fi
REPOSITORY_NAME="${REPOSITORY_URI#*/}"

if [[ "${FALLBACK_REGION}" != "${REGION}" ]]; then
  if ! FALLBACK_REPOSITORY_URI="$(
    aws cloudformation describe-stacks \
      --region "${FALLBACK_REGION}" \
      --stack-name "${FALLBACK_FOUNDATION_STACK}" \
      --query "Stacks[0].Outputs[?OutputKey=='RepositoryUri'].OutputValue | [0]" \
      --output text
  )"; then
    echo "Deploy ${FALLBACK_FOUNDATION_STACK} before pushing the first replicated image." >&2
    exit 1
  fi
  if [[ -z "${FALLBACK_REPOSITORY_URI}" || "${FALLBACK_REPOSITORY_URI}" == "None" ]]; then
    echo "No RepositoryUri output exists on ${FALLBACK_FOUNDATION_STACK}." >&2
    echo "Deploy the updated fallback foundation stack before building the image." >&2
    exit 1
  fi
  FALLBACK_REPOSITORY_NAME="${FALLBACK_REPOSITORY_URI#*/}"
  if [[ "${FALLBACK_REPOSITORY_NAME}" != "${REPOSITORY_NAME}" ]]; then
    echo "ECR replication requires matching repository names, but found:" >&2
    echo "  ${REGION}: ${REPOSITORY_NAME}" >&2
    echo "  ${FALLBACK_REGION}: ${FALLBACK_REPOSITORY_NAME}" >&2
    echo "Deploy the updated storage and fallback foundation stacks first." >&2
    exit 1
  fi
  python3 "${ROOT_DIR}/jobs/ecr_replication.py" \
    --source-region "${REGION}" \
    --destination-region "${FALLBACK_REGION}" \
    --repository-name "${REPOSITORY_NAME}"
fi

SOURCE_HASH="$(
  cd "${ROOT_DIR}"
  python3 - <<'PY'
import hashlib
import pathlib
import subprocess

root = pathlib.Path.cwd()
paths = subprocess.check_output(
    ["git", "ls-files", "-co", "--exclude-standard", "-z"]
).decode().split("\0")
digest = hashlib.sha256()
for name in sorted(filter(None, paths)):
    path = root / name
    if path.is_file():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
print(digest.hexdigest()[:12])
PY
)"
GIT_SHA="$(git -C "${ROOT_DIR}" rev-parse --short=12 HEAD)"
IMAGE_TAG="${IMAGE_TAG:-${GIT_SHA}-${SOURCE_HASH}}"
IMAGE_URI="${REPOSITORY_URI}:${IMAGE_TAG}"
REGISTRY="${REPOSITORY_URI%%/*}"

if image_exists "${REGION}" "${REPOSITORY_NAME}" "${IMAGE_TAG}"; then
  echo "Image already exists; skipped build and push: ${IMAGE_URI}"
else
  status=$?
  if [[ "${status}" -ne 1 ]]; then
    exit "${status}"
  fi
  aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${REGISTRY}"
  docker buildx build \
    --platform linux/amd64 \
    --push \
    --tag "${IMAGE_URI}" \
    "${ROOT_DIR}"
  echo "Pushed ${IMAGE_URI}"
fi

if [[ "${FALLBACK_REGION}" != "${REGION}" ]]; then
  deadline=$((SECONDS + REPLICATION_WAIT_SECONDS))
  echo "Waiting for ECR to replicate ${IMAGE_TAG} to ${FALLBACK_REGION}..."
  while true; do
    if image_exists "${FALLBACK_REGION}" "${FALLBACK_REPOSITORY_NAME}" "${IMAGE_TAG}"; then
      echo "Replicated ${FALLBACK_REPOSITORY_URI}:${IMAGE_TAG}"
      break
    else
      status=$?
      if [[ "${status}" -ne 1 ]]; then
        exit "${status}"
      fi
    fi
    if (( SECONDS >= deadline )); then
      echo "Image did not replicate to ${FALLBACK_REGION} within ${REPLICATION_WAIT_SECONDS}s." >&2
      echo "Verify the GeneralsTrainingStorage ECR replication configuration." >&2
      exit 1
    fi
    sleep 5
  done
fi

echo "Image tag: ${IMAGE_TAG}"
echo "Deploy Batch with:"
if [[ "${FALLBACK_REGION}" != "${REGION}" ]]; then
  echo "  cd infra/aws && ./cdk.sh deploy GeneralsTrainingBatch GeneralsTrainingBatchUsWest2 --require-approval never -c enableWest2=true -c artifactBucketName=${ARTIFACT_BUCKET} --parameters GeneralsTrainingBatch:ImageTag=${IMAGE_TAG} --parameters GeneralsTrainingBatchUsWest2:ImageTag=${IMAGE_TAG}"
else
  echo "  cd infra/aws && ./cdk.sh deploy GeneralsTrainingBatch --require-approval never --parameters GeneralsTrainingBatch:ImageTag=${IMAGE_TAG}"
fi
