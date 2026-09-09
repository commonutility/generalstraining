# AWS Batch GPU jobs

> Do not directly provision EC2 instances for normal training jobs. Use the AWS Batch job interface.

The `GeneralsTrainingBatch` stack owns managed On-Demand and Spot GPU compute
environments, dedicated queues, a job definition, IAM roles, and CloudWatch
logging. Batch launches capacity when jobs are runnable and returns to
`minvCpus=0` when each queue is idle.

The stack reuses:

- the retained S3 bucket from `GeneralsTrainingStorage`;
- the dedicated `ml-lab-vpc`;
- its public subnets in all six `us-east-1` Availability Zones; and
- the no-ingress `ml-lab-gpu-sg` security group.

It does not use the unrelated `axis-*` production network.

## Deploy the On-Demand system

Use the active AWS CLI/SSO profile and region. Do not place credentials in the
repository.

```bash
cd infra/aws
npm ci
npm run build
npm test

# Adds the retained, stable-name primary ECR repository.
./cdk.sh deploy GeneralsTrainingStorage --require-approval never

# Create the same-named destination repository before the first image push.
ARTIFACT_BUCKET="$(aws cloudformation describe-stacks \
  --region us-east-1 \
  --stack-name GeneralsTrainingStorage \
  --query 'Stacks[0].Outputs[?OutputKey==`CheckpointBucketName`].OutputValue' \
  --output text)"
./cdk.sh deploy GeneralsTrainingWest2Foundation \
  --require-approval never \
  -c enableWest2=true \
  -c artifactBucketName="${ARTIFACT_BUCKET}"

cd ../..
./jobs/build_and_push.sh
```

`build_and_push.sh` hashes the source, reuses an existing immutable primary
image when possible, idempotently adds the filtered ECR replication rule while
preserving unrelated registry rules, pushes a missing image once to
`us-east-1`, and waits for ECR to replicate it server-side to `us-west-2`. Use
its printed immutable tag to deploy the Batch stack:

```bash
cd infra/aws
./cdk.sh deploy GeneralsTrainingBatch \
  --require-approval never \
  --parameters GeneralsTrainingBatch:ImageTag=<printed-image-tag>
```

Review `./cdk.sh diff` before either deployment. The wrapper exports temporary
credentials from the active CLI/SSO profile only for the CDK process.

The image tag is a required CloudFormation parameter with no fake default.
After the first parameterized deployment, CDK's default
`--previous-parameters` behavior preserves the deployed tag when the parameter
is omitted. If `--no-previous-parameters` is used, the image tag must be
supplied again.

The first storage deployment after adopting the stable `generals-training`
repository name replaces the old generated-name primary repository. The old
repository is retained so existing job-definition revisions remain pullable;
remove it only after confirming no active or rollback revision refers to it.

## Use the restricted job profile

The Batch stack creates `generals-training-job-submitter`, trusted only by this
account's AWS IAM Identity Center administrator role. Configure a local
role-assumption profile after deployment:

```bash
ROLE_ARN="$(aws cloudformation describe-stacks \
  --stack-name GeneralsTrainingBatch \
  --query 'Stacks[0].Outputs[?OutputKey==`JobSubmitterRoleArn`].OutputValue' \
  --output text)"
aws configure set role_arn "${ROLE_ARN}" --profile generals-jobs
aws configure set source_profile "${AWS_PROFILE:-default}" --profile generals-jobs
aws configure set region us-east-1 --profile generals-jobs
```

Use `python jobs/cli.py --profile generals-jobs ...` for normal operations.
This role can submit only the project job definition to its project queues, inspect
and stop jobs, read CloudWatch output, access the experiment S3 prefix, and
push/pull the project ECR repository. It has no direct EC2, VPC, security-group,
or IAM administration permissions. Continue to use the administrator profile
only for reviewed infrastructure deployments.

## Launch a job

```bash
python jobs/cli.py --profile generals-jobs submit \
  --name spr-pretrain-001 \
  --command "python scripts/train_ppo.py --config configs/experiments/s_budget.yaml"
```

The command returns after Batch accepts the job and prints the experiment ID,
job ID, region, immutable image, command, and S3 destination.

Optional controls:

```text
--experiment-id NAME
--workload-class small_gpu|high_memory_gpu
--compute ondemand|spot
--cpus 4
--memory 15000
--gpu 1
--attempts 1
--timeout-seconds 7200
--env KEY=VALUE
--resume-from s3://bucket/generals/experiments/old-run/checkpoints/run/model.eqx
```

For `scripts/train_ppo.py`, `--resume-from` downloads the object and supplies it as
`--init_checkpoint`. A custom command can place `{resume_from}` where the local
downloaded path belongs. `--resume-ema-from` downloads a second object (the EMA
weights saved beside each full milestone) and substitutes it for `{resume_ema}`.

## Resume training from milestone checkpoints

Full milestone checkpoints (`<run>_<iter>.eqx`) contain the network and Adam
optimizer state, so training can continue from them. Resume a whole seed matrix
with per-seed checkpoint URIs; the matching `_ema_` URI is derived
automatically:

```bash
python jobs/cli.py --profile generals-jobs submit-matrix \
  --run-prefix exp8k \
  --config configs/experiments/L_7d_gae90_8k.yaml \
  --seeds 44,45,46 \
  --num-iters 6000 --save-at 4000 8000 \
  --iteration-offset 2000 \
  --resume-checkpoints \
    44=s3://<bucket>/generals/experiments/<exp44>/checkpoints/<run44>/<run44>_2000.eqx \
    45=s3://<bucket>/generals/experiments/<exp45>/checkpoints/<run45>/<run45>_2000.eqx \
    46=s3://<bucket>/generals/experiments/<exp46>/checkpoints/<run46>/<run46>_2000.eqx
```

`--iteration-offset` is the global iteration count already completed by the
checkpoints. It shifts the entropy/gamma schedules to continue where the source
run stopped (the LR schedule continues automatically through the restored Adam
step count), and it makes logged iterations, `train/env_interactions`,
checkpoint filenames, and `--save-at` milestones global. In the example above,
`--num-iters 6000` runs global iterations 2,001-8,000 and `--save-at 4000 8000`
writes `<run>_4000.eqx` and `<run>_8000.eqx`.

Two aspects intentionally do not resume: the RNG stream restarts from the seed,
and the curriculum restarts at stage 0 and re-advances through its win-rate
gates (about 200 iterations at `eval_every: 50` when the agent passes each gate
on the first eval).

## Inspect a job

```bash
python jobs/cli.py --profile generals-jobs status <job-id>
python jobs/cli.py --profile generals-jobs logs <job-id>
python jobs/cli.py --profile generals-jobs logs <job-id> --follow
```

`status` includes Batch, container, and attempt reasons so capacity, image,
IAM, out-of-memory, and application failures can be distinguished. Normal
debugging uses CloudWatch logs and does not require SSH.

## Stop a job

```bash
python jobs/cli.py --profile generals-jobs cancel <job-id>
```

Queued jobs are cancelled and running jobs are terminated through the Batch
API. Do not terminate the underlying ECS container instance.

## Persistent experiment layout

The container wrapper writes to:

```text
s3://<bucket>/generals/experiments/<experiment-id>/
    config/job.json
    config/<run-name>.yaml
    checkpoints/
    metrics/result.json
    artifacts/
```

It uploads stable checkpoints periodically and performs a final upload even
when the training command fails. Exit code `75` means durable artifact I/O
failed and is eligible for a limited Batch retry. Application failures are not
retried indefinitely.

## Capacity and workload classes

`small_gpu` requires one GPU, 4 vCPUs, at most 15,000 MiB container memory, and
24 GB GPU memory. Compatible capacity:

- `g6e.xlarge`: NVIDIA L40S, 48 GB GPU memory, 32 GiB system memory;
- `g6.xlarge`: NVIDIA L4, 24 GB GPU memory, 16 GiB system memory;
- `g5.xlarge`: NVIDIA A10G, 24 GB GPU memory, 16 GiB system memory.

These types are offered across the configured Availability Zones. The small-GPU
environments are capped at 4 vCPUs; current account quotas are 8 On-Demand and
4 Spot G/VT vCPUs.

`high_memory_gpu` targets 80 GB-class GPUs. The On-Demand environment offers
`g7e.2xlarge` (one RTX PRO Server 6000, 96 GB GPU memory, 8 vCPUs),
`p5.4xlarge` (one H100, 80 GB GPU memory, 16 vCPUs), and `p4de.24xlarge`
(eight A100s, 80 GB GPU memory each, 96 vCPUs); the Spot environment offers
the two P types. `BEST_FIT_PROGRESSIVE` tries the smallest fit first. The P
types run on the 128-vCPU P-instance quota; the G7e needs the 8-vCPU G/VT
quota approved on 2026-08-21 (EC2 counts an instance type's full vCPUs against
the quota even when a launch template trims active vCPUs with `CpuOptions`).
The one-iteration full-shape probe succeeded on a `g7e.2xlarge` in
`us-east-1b` after H100 and A100 capacity was exhausted in all six zones.
The `GeneralsTrainingNetworkExtension` stack adds `us-east-1c` through `1f`
fallback subnets because the original zones lacked On-Demand H100 capacity.

The unmodified 512-environment `L_7d_gae90_8k` workload exhausted a 48 GB L40S.
Gate the full run on a one-iteration 80GB-or-larger probe:

```bash
python jobs/cli.py --profile generals-jobs submit \
  --name full-shape-memory-probe \
  --workload-class high_memory_gpu \
  --command "python scripts/train_ppo.py --config configs/experiments/L_7d_gae90_8k.yaml --num_iters 1 --save_at 1 --run_name full_shape_80gb_probe"
```

High-memory jobs default to one attempt and a two-hour hard timeout. Batch
terminates a timed-out container, and the managed compute environment has
`minvCpus=0`, so it scales back to zero after every terminal job state. Longer
runs must opt into a larger bounded value with `--timeout-seconds`; retries must
be explicitly enabled with `--attempts`.

The On-Demand allocation strategy is `BEST_FIT_PROGRESSIVE`. Both Spot
environments use AWS Batch's recommended `SPOT_PRICE_CAPACITY_OPTIMIZED`
strategy; the high-memory Spot environment offers `p5.4xlarge` and
`p4de.24xlarge`.

## Spot jobs

Spot is opt-in for both workload classes:

```bash
python jobs/cli.py --profile generals-jobs submit \
  --name interruptible-training \
  --compute spot \
  --command "python scripts/train_ppo.py --config configs/example.yaml"
```

Use Spot only for commands that write checkpoints to the experiment directory
and can resume with `--resume-from`. EC2 can reclaim Spot capacity with two
minutes of notice. Batch may retry an infrastructure interruption according to
the job definition, while stable checkpoints remain in S3. The one-iteration
H100 probe uses one attempt and a two-hour hard timeout.

## Acceptance tests

GPU smoke test:

```bash
python jobs/cli.py --profile generals-jobs submit \
  --name gpu-smoke-test \
  --command "python jobs/gpu_smoke.py"
```

Short real training test:

```bash
python jobs/cli.py --profile generals-jobs submit \
  --name ppo-smoke-test \
  --command "python scripts/train_ppo.py --config configs/smoke/L_7d_gae90_cpu.yaml --run_name batch_ppo_smoke"
```

After both pass, verify the experiment S3 prefix, CloudWatch logs, Batch job
status, and that the compute environment returns to zero instances.

## Regional fallback

`us-east-1` remains preferred. The `us-west-2` fallback stacks create a
four-AZ public VPC with no NAT gateway, an immutable regional ECR repository,
and the same four scale-to-zero Batch queues. They reuse the primary S3 bucket
and extend the existing `generals-training-job-submitter` role instead of
creating another operator identity.

The fallback costs about `$0.29/month` while idle for the current 2.876 GB ECR
image. Batch and the NAT-free VPC have no idle hourly charge. Current
On-Demand prices are `$0.8048-$1.861/hour` for the small GPU class and
`$3.36312-$6.88/hour` for high-memory capacity, excluding EBS, public IPv4,
logs, and cross-region S3 transfer.

The `us-west-2` G/VT and P On-Demand/Spot quota requests were approved on
2026-08-21. To recreate or update the regional stacks:

```bash
cd infra/aws
./cdk.sh deploy GeneralsTrainingStorage --require-approval never
cd ../..

ARTIFACT_BUCKET="$(aws cloudformation describe-stacks \
  --region us-east-1 \
  --stack-name GeneralsTrainingStorage \
  --query 'Stacks[0].Outputs[?OutputKey==`CheckpointBucketName`].OutputValue' \
  --output text)"

cd infra/aws
./cdk.sh deploy GeneralsTrainingWest2Foundation \
  --require-approval never \
  -c enableWest2=true \
  -c artifactBucketName="${ARTIFACT_BUCKET}"

cd ../..
./jobs/build_and_push.sh

# Use the immutable tag printed by build_and_push.sh.
cd infra/aws
./cdk.sh deploy GeneralsTrainingBatch GeneralsTrainingBatchUsWest2 \
  --require-approval never \
  -c enableWest2=true \
  -c artifactBucketName="${ARTIFACT_BUCKET}" \
  --parameters GeneralsTrainingBatch:ImageTag=<printed-image-tag> \
  --parameters GeneralsTrainingBatchUsWest2:ImageTag=<printed-image-tag>
```

A second fallback region, `us-east-2`, mirrors the `us-west-2` design with two
differences: its Batch job definition pulls the training image cross-region
from the primary `us-east-1` repository (no regional image copy is needed),
and its high-memory compute environments list only `g7e.2xlarge` and
`p5.4xlarge` because `p4de.24xlarge` is not offered in `us-east-2`. Deploy it
with `-c enableEast2=true`, the `artifactBucketName` context, a
`primaryRepositoryArn` context pointing at the `us-east-1` training
repository ARN, and `--parameters GeneralsTrainingBatchUsEast2:ImageTag=<tag>`.
The G On-Demand quota there was approved at 32 vCPUs (four concurrent
`g7e.2xlarge` jobs) on 2026-08-21.

ECR replication configuration is registry-wide, not repository-owned.
`build_and_push.sh` therefore preserves existing rules and adds a
`generals-training` prefix rule from `us-east-1` to `us-west-2` only when it is
not already covered. It never uploads the image from the workstation twice and
verifies that the destination copy arrives before printing the deployment
command. ECR only replicates new image pushes, so if a primary tag predates the
replication rule and is absent in the fallback repository, build a new source
revision rather than trying to overwrite the immutable tag.

Submit directly to the fallback region when explicitly requested:

```bash
python jobs/cli.py \
  --profile generals-jobs \
  --region us-west-2 \
  submit --name west-gpu-smoke --command "python jobs/gpu_smoke.py"
```

For bounded automatic fallback, opt in on submission:

```bash
python jobs/cli.py --profile generals-jobs submit \
  --name resilient-training \
  --fallback-region us-west-2 \
  --fallback-after-seconds 900 \
  --command "python scripts/train_ppo.py --config configs/example.yaml"
```

The launcher does not duplicate jobs. It keeps the preferred job when it
starts or when older jobs explain the queue delay. Otherwise it cancels a
stalled preferred job, waits for terminal confirmation, and only then submits
the same globally unique experiment ID to `us-west-2`. Results retain the same
primary S3 prefix. Use `--region us-west-2` with `status`, `logs`, and `cancel`
for a fallback job ID.

## When infrastructure changes are appropriate

Use the infrastructure administrator credentials only for reviewed changes to
the CDK stacks, quota increases, image/job-definition releases, or recovery of
a failed CloudFormation deployment. Normal experiments must not create or
delete VPCs, subnets, security groups, IAM roles, or standalone EC2 instances.

Regional infrastructure changes still require the administrator profile.
Normal experiments, including regional fallback, use only the job CLI and the
restricted `generals-jobs` profile.
