#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib/core';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import {
  TrainingBatchStack,
  TrainingNetworkExtensionStack,
  TrainingStorageStack,
} from '../lib/aws-stack';
import { TrainingRegionalFoundationStack } from '../lib/aws-regional-stack';

const app = new cdk.App();
const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION ?? 'us-east-1',
};

const storage = new TrainingStorageStack(app, 'GeneralsTrainingStorage', {
  env,
  terminationProtection: true,
  description: 'Durable storage for Generals.io PPO checkpoints and training logs',
});

const baseAvailabilityZones = app.node.getContext('gpuAvailabilityZones') as string[];
const basePublicSubnetIds = app.node.getContext('gpuPublicSubnetIds') as string[];
const basePublicRouteTableIds = app.node.getContext('gpuPublicSubnetRouteTableIds') as string[];
const networkExtension = new TrainingNetworkExtensionStack(app, 'GeneralsTrainingNetworkExtension', {
  env,
  vpcId: app.node.getContext('gpuVpcId'),
  publicRouteTableId: basePublicRouteTableIds[0],
  description: 'Additional public subnets for H100 capacity fallback',
});

new TrainingBatchStack(app, 'GeneralsTrainingBatch', {
  env,
  checkpointBucket: storage.checkpointBucket,
  trainingRepository: storage.trainingRepository,
  maxvCpus: Number(app.node.tryGetContext('maxvCpus') ?? 4),
  spotMaxvCpus: Number(app.node.tryGetContext('spotMaxvCpus') ?? 4),
  highMemoryMaxvCpus: Number(app.node.tryGetContext('highMemoryMaxvCpus') ?? 96),
  highMemorySpotMaxvCpus: Number(app.node.tryGetContext('highMemorySpotMaxvCpus') ?? 96),
  vpcId: app.node.getContext('gpuVpcId'),
  availabilityZones: [...baseAvailabilityZones, ...networkExtension.capacityAvailabilityZones],
  publicSubnetIds: [...basePublicSubnetIds, ...networkExtension.publicSubnetIds],
  publicSubnetRouteTableIds: [
    ...basePublicRouteTableIds,
    ...networkExtension.publicSubnetIds.map(() => basePublicRouteTableIds[0]),
  ],
  securityGroupId: app.node.getContext('gpuSecurityGroupId'),
  description: 'Scale-to-zero AWS Batch GPU training environment',
});

if (String(app.node.tryGetContext('enableWest2')).toLowerCase() === 'true') {
  const artifactBucketName = app.node.tryGetContext('artifactBucketName');
  if (!artifactBucketName) {
    throw new Error('artifactBucketName context is required when enableWest2=true');
  }

  const westEnv = {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: 'us-west-2',
  };
  const westFoundation = new TrainingRegionalFoundationStack(
    app,
    'GeneralsTrainingWest2Foundation',
    {
      env: westEnv,
      artifactBucketName,
      vpcCidr: '10.41.0.0/16',
      terminationProtection: true,
      description: 'Regional VPC and ECR repository for Generals Batch fallback',
    },
  );

  new TrainingBatchStack(app, 'GeneralsTrainingBatchUsWest2', {
    env: westEnv,
    checkpointBucket: westFoundation.artifactBucket,
    trainingRepository: westFoundation.trainingRepository,
    maxvCpus: Number(app.node.tryGetContext('westMaxvCpus') ?? 4),
    spotMaxvCpus: Number(app.node.tryGetContext('westSpotMaxvCpus') ?? 4),
    highMemoryMaxvCpus: Number(app.node.tryGetContext('westHighMemoryMaxvCpus') ?? 16),
    highMemorySpotMaxvCpus: Number(
      app.node.tryGetContext('westHighMemorySpotMaxvCpus') ?? 16,
    ),
    vpc: westFoundation.vpc,
    securityGroup: westFoundation.securityGroup,
    submitterRoleName: 'generals-training-job-submitter',
    description: 'Scale-to-zero AWS Batch GPU fallback environment in us-west-2',
  });
}

if (String(app.node.tryGetContext('enableEast2')).toLowerCase() === 'true') {
  const artifactBucketName = app.node.tryGetContext('artifactBucketName');
  if (!artifactBucketName) {
    throw new Error('artifactBucketName context is required when enableEast2=true');
  }
  const primaryRepositoryArn = app.node.tryGetContext('primaryRepositoryArn');
  if (!primaryRepositoryArn) {
    throw new Error('primaryRepositoryArn context is required when enableEast2=true');
  }

  const east2Env = {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: 'us-east-2',
  };
  const east2Foundation = new TrainingRegionalFoundationStack(
    app,
    'GeneralsTrainingEast2Foundation',
    {
      env: east2Env,
      artifactBucketName,
      vpcCidr: '10.42.0.0/16',
      terminationProtection: true,
      description: 'Regional VPC and ECR repository for Generals Batch fallback in us-east-2',
    },
  );

  // Pull the training image cross-region from the primary us-east-1 repository
  // so this fallback region never waits on a multi-gigabyte image re-upload.
  const primaryRepository = ecr.Repository.fromRepositoryAttributes(
    east2Foundation,
    'PrimaryTrainingRepository',
    {
      repositoryArn: primaryRepositoryArn,
      repositoryName: primaryRepositoryArn.split('/').pop()!,
    },
  );

  new TrainingBatchStack(app, 'GeneralsTrainingBatchUsEast2', {
    env: east2Env,
    checkpointBucket: east2Foundation.artifactBucket,
    trainingRepository: primaryRepository,
    maxvCpus: Number(app.node.tryGetContext('east2MaxvCpus') ?? 4),
    spotMaxvCpus: Number(app.node.tryGetContext('east2SpotMaxvCpus') ?? 4),
    highMemoryMaxvCpus: Number(app.node.tryGetContext('east2HighMemoryMaxvCpus') ?? 32),
    highMemorySpotMaxvCpus: Number(
      app.node.tryGetContext('east2HighMemorySpotMaxvCpus') ?? 16,
    ),
    vpc: east2Foundation.vpc,
    securityGroup: east2Foundation.securityGroup,
    // us-east-2 does not offer p4de.24xlarge; Batch rejects unknown regional types.
    highMemoryInstanceTypes: ['g7e.2xlarge', 'p5.4xlarge'],
    submitterRoleName: 'generals-training-job-submitter',
    description: 'Scale-to-zero AWS Batch GPU fallback environment in us-east-2',
  });
}

cdk.Tags.of(app).add('Project', 'GeneralsTraining');
cdk.Tags.of(app).add('ManagedBy', 'CDK');
