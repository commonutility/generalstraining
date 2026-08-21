# Generals training on AWS Batch

This CDK app provisions:

- A retained, encrypted, private S3 bucket for experiment state.
- Retained, immutable `generals-training` ECR repositories; the release helper
  configures filtered, server-side replication from `us-east-1` to
  `us-west-2` without replacing unrelated registry rules.
- Managed On-Demand and Spot AWS Batch GPU environments with `minvCpus=0`.
- Dedicated standard, Spot, and high-memory queues with a one-GPU job definition.
- A retained CloudWatch log group.

The Batch stack reuses the existing two-AZ `ml-lab-vpc` and its no-ingress GPU
security group. It does not create standalone training instances, NAT gateways,
or a new VPC.

With `-c enableWest2=true`, the app also exposes
`GeneralsTrainingWest2Foundation` and `GeneralsTrainingBatchUsWest2`. The
foundation creates a NAT-free four-AZ fallback VPC and regional ECR repository;
the Batch stack reuses the primary S3 bucket and submitter role. See the main
AWS jobs guide for the quota gate and deployment command. Batch image tags are
required CloudFormation parameters; omitting one on a later deploy preserves
the previously deployed value instead of selecting a placeholder image.

```bash
npm ci
npm run build
npm test
./cdk.sh synth
./cdk.sh diff
```

See `../../docs/aws-jobs.md` for the deployment order, image build, job CLI,
acceptance tests, and operational boundaries.
