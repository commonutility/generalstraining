import * as cdk from 'aws-cdk-lib/core';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Match, Template } from 'aws-cdk-lib/assertions';
import {
  TrainingBatchStack,
  TrainingNetworkExtensionStack,
  TrainingStorageStack,
} from '../lib/aws-stack';
import { TrainingRegionalFoundationStack } from '../lib/aws-regional-stack';

const app = new cdk.App();
const testEnv = { account: '111111111111', region: 'us-east-1' };
const storage = new TrainingStorageStack(app, 'TestStorage', { env: testEnv });
const network = new cdk.Stack(app, 'TestNetwork', { env: testEnv });
const vpc = new ec2.Vpc(network, 'Vpc', {
  maxAzs: 2,
  natGateways: 0,
  subnetConfiguration: [{
    name: 'public',
    subnetType: ec2.SubnetType.PUBLIC,
  }],
});
const securityGroup = new ec2.SecurityGroup(network, 'SecurityGroup', { vpc });
const networkExtension = new TrainingNetworkExtensionStack(app, 'TestNetworkExtension', {
  vpcId: 'vpc-0123456789abcdef0',
  publicRouteTableId: 'rtb-0123456789abcdef0',
  env: testEnv,
});
const batch = new TrainingBatchStack(app, 'TestBatch', {
  checkpointBucket: storage.checkpointBucket,
  trainingRepository: storage.trainingRepository,
  vpc,
  securityGroup,
  env: testEnv,
});
const regionalApp = new cdk.App({
  context: {
    'availability-zones:account=111111111111:region=us-west-2': [
      'us-west-2a',
      'us-west-2b',
      'us-west-2c',
      'us-west-2d',
    ],
  },
});
const regionalEnv = { account: '111111111111', region: 'us-west-2' };
const regionalFoundation = new TrainingRegionalFoundationStack(
  regionalApp,
  'TestWestFoundation',
  {
    artifactBucketName: 'primary-artifact-bucket',
    env: regionalEnv,
  },
);
const regionalBatch = new TrainingBatchStack(regionalApp, 'TestWestBatch', {
  checkpointBucket: regionalFoundation.artifactBucket,
  trainingRepository: regionalFoundation.trainingRepository,
  vpc: regionalFoundation.vpc,
  securityGroup: regionalFoundation.securityGroup,
  submitterRoleName: 'generals-training-job-submitter',
  env: regionalEnv,
});

test('fallback foundation creates retained regional ECR and four public subnets without NAT', () => {
  const template = Template.fromStack(regionalFoundation);

  template.hasResourceProperties('AWS::EC2::VPC', {
    CidrBlock: '10.41.0.0/16',
    EnableDnsHostnames: true,
    EnableDnsSupport: true,
  });
  template.resourceCountIs('AWS::EC2::Subnet', 4);
  template.resourceCountIs('AWS::EC2::NatGateway', 0);
  template.hasResourceProperties('AWS::ECR::Repository', {
    RepositoryName: 'generals-training',
    ImageScanningConfiguration: { ScanOnPush: true },
    ImageTagMutability: 'IMMUTABLE',
  });
  template.hasResource('AWS::ECR::Repository', {
    DeletionPolicy: 'Retain',
    UpdateReplacePolicy: 'Retain',
  });
});

test('fallback Batch stack extends the existing submitter role instead of duplicating it', () => {
  const template = Template.fromStack(regionalBatch);
  const namedRoles = Object.values(template.findResources('AWS::IAM::Role')).filter(
    resource => resource.Properties?.RoleName === 'generals-training-job-submitter',
  );

  expect(namedRoles).toHaveLength(0);
  template.hasResourceProperties('AWS::IAM::Policy', {
    Roles: ['generals-training-job-submitter'],
    PolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: 'batch:SubmitJob',
          Effect: 'Allow',
        }),
      ]),
    },
  });
});

test('capacity fallback adds public subnets in all remaining availability zones', () => {
  const template = Template.fromStack(networkExtension);

  template.resourceCountIs('AWS::EC2::Subnet', 4);
  template.resourceCountIs('AWS::EC2::SubnetRouteTableAssociation', 4);
  template.hasResourceProperties('AWS::EC2::Subnet', {
    AvailabilityZone: 'us-east-1c',
    CidrBlock: '10.40.3.0/24',
    MapPublicIpOnLaunch: true,
    VpcId: 'vpc-0123456789abcdef0',
  });
  template.hasResourceProperties('AWS::EC2::Subnet', {
    AvailabilityZone: 'us-east-1d',
    CidrBlock: '10.40.4.0/24',
    MapPublicIpOnLaunch: true,
    VpcId: 'vpc-0123456789abcdef0',
  });
  template.hasResourceProperties('AWS::EC2::Subnet', {
    AvailabilityZone: 'us-east-1e',
    CidrBlock: '10.40.5.0/24',
    MapPublicIpOnLaunch: true,
    VpcId: 'vpc-0123456789abcdef0',
  });
  template.hasResourceProperties('AWS::EC2::Subnet', {
    AvailabilityZone: 'us-east-1f',
    CidrBlock: '10.40.6.0/24',
    MapPublicIpOnLaunch: true,
    VpcId: 'vpc-0123456789abcdef0',
  });
});

test('checkpoint bucket is private, encrypted, versioned, and retained', () => {
  const template = Template.fromStack(storage);

  template.hasResourceProperties('AWS::S3::Bucket', {
    BucketEncryption: {
      ServerSideEncryptionConfiguration: [
        {
          ServerSideEncryptionByDefault: {
            SSEAlgorithm: 'AES256',
          },
        },
      ],
    },
    PublicAccessBlockConfiguration: {
      BlockPublicAcls: true,
      BlockPublicPolicy: true,
      IgnorePublicAcls: true,
      RestrictPublicBuckets: true,
    },
    VersioningConfiguration: { Status: 'Enabled' },
  });
  template.hasResource('AWS::S3::Bucket', {
    DeletionPolicy: 'Retain',
    UpdateReplacePolicy: 'Retain',
  });
});

test('ECR repository is immutable, scanned, and retained', () => {
  const template = Template.fromStack(storage);

  template.hasResourceProperties('AWS::ECR::Repository', {
    RepositoryName: 'generals-training',
    ImageScanningConfiguration: { ScanOnPush: true },
    ImageTagMutability: 'IMMUTABLE',
  });
  template.hasResource('AWS::ECR::Repository', {
    DeletionPolicy: 'Retain',
    UpdateReplacePolicy: 'Retain',
  });
});

test('Batch requires an image tag parameter with no unsafe default', () => {
  const template = Template.fromStack(batch).toJSON();

  expect(template.Parameters.ImageTag).toMatchObject({
    Type: 'String',
    AllowedPattern: '[A-Za-z0-9_][A-Za-z0-9_.-]{0,299}',
  });
  expect(template.Parameters.ImageTag).not.toHaveProperty('Default');
  expect(template.Outputs.ImageTag.Value).toEqual({ Ref: 'ImageTag' });
});

test('Batch compute scales to zero across compatible GPU types', () => {
  const template = Template.fromStack(batch);

  template.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
    ComputeEnvironmentName: 'generals-gpu-ondemand',
    ComputeResources: Match.objectLike({
      AllocationStrategy: 'BEST_FIT_PROGRESSIVE',
      InstanceTypes: ['g6e.xlarge', 'g6.xlarge', 'g5.xlarge'],
      MinvCpus: 0,
      MaxvCpus: 4,
      Type: 'EC2',
    }),
  });
  template.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
    ComputeEnvironmentName: 'generals-high-memory-gpu-ondemand',
    ComputeResources: Match.objectLike({
      AllocationStrategy: 'BEST_FIT_PROGRESSIVE',
      InstanceTypes: ['g7e.2xlarge', 'p5.4xlarge', 'p4de.24xlarge'],
      MinvCpus: 0,
      MaxvCpus: 96,
      Type: 'EC2',
    }),
  });
  template.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
    ComputeEnvironmentName: 'generals-gpu-spot',
    ComputeResources: Match.objectLike({
      AllocationStrategy: 'SPOT_PRICE_CAPACITY_OPTIMIZED',
      InstanceTypes: ['g6e.xlarge', 'g6.xlarge', 'g5.xlarge'],
      MinvCpus: 0,
      MaxvCpus: 4,
      Type: 'SPOT',
    }),
  });
  template.hasResourceProperties('AWS::Batch::ComputeEnvironment', {
    ComputeEnvironmentName: 'generals-high-memory-gpu-spot',
    ComputeResources: Match.objectLike({
      AllocationStrategy: 'SPOT_PRICE_CAPACITY_OPTIMIZED',
      InstanceTypes: ['p5.4xlarge', 'p4de.24xlarge'],
      MinvCpus: 0,
      MaxvCpus: 96,
      Type: 'SPOT',
    }),
  });
  template.hasResourceProperties('AWS::Batch::JobQueue', {
    JobQueueName: 'gpu-training-high-memory',
    State: 'ENABLED',
  });
  template.hasResourceProperties('AWS::Batch::JobQueue', {
    JobQueueName: 'gpu-training-spot',
    State: 'ENABLED',
  });
  template.hasResourceProperties('AWS::Batch::JobQueue', {
    JobQueueName: 'gpu-training-high-memory-spot',
    State: 'ENABLED',
  });
  template.resourceCountIs('AWS::Batch::ComputeEnvironment', 4);
  template.hasResourceProperties('AWS::EC2::LaunchTemplate', {
    LaunchTemplateData: Match.objectLike({
      MetadataOptions: {
        HttpTokens: 'required',
      },
      BlockDeviceMappings: Match.arrayWith([
        Match.objectLike({
          Ebs: Match.objectLike({
            Encrypted: true,
            VolumeSize: 100,
            VolumeType: 'gp3',
          }),
        }),
      ]),
    }),
  });
});

test('Batch job requests one GPU and sends logs to CloudWatch', () => {
  const template = Template.fromStack(batch);

  template.hasResourceProperties('AWS::Batch::JobDefinition', {
    ContainerProperties: Match.objectLike({
      ResourceRequirements: Match.arrayWith([{ Type: 'VCPU', Value: '4' }]),
      LogConfiguration: Match.objectLike({
        LogDriver: 'awslogs',
      }),
    }),
    RetryStrategy: Match.objectLike({
      Attempts: 3,
    }),
  });
  template.hasResourceProperties('AWS::Batch::JobDefinition', {
    ContainerProperties: Match.objectLike({
      ResourceRequirements: Match.arrayWith([{ Type: 'MEMORY', Value: '15000' }]),
    }),
  });
  template.hasResourceProperties('AWS::Batch::JobDefinition', {
    ContainerProperties: Match.objectLike({
      ResourceRequirements: Match.arrayWith([{ Type: 'GPU', Value: '1' }]),
    }),
  });
  template.hasResourceProperties('AWS::Logs::LogGroup', {
    LogGroupName: '/aws/batch/generals-training',
    RetentionInDays: 30,
  });
});

test('job submitter role is limited to the Batch job interface', () => {
  const template = Template.fromStack(batch);

  template.hasResourceProperties('AWS::IAM::Role', {
    RoleName: 'generals-training-job-submitter',
  });
  template.hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: 'batch:SubmitJob',
          Effect: 'Allow',
        }),
        Match.objectLike({
          Action: 'batch:TagResource',
          Effect: 'Allow',
        }),
        Match.objectLike({
          Action: ['batch:DescribeJobs', 'batch:ListJobs'],
          Effect: 'Allow',
          Resource: '*',
        }),
      ]),
    },
  });
});
