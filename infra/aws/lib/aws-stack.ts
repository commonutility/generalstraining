import * as batch from 'aws-cdk-lib/aws-batch';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cdk from 'aws-cdk-lib/core';
import { Construct } from 'constructs';

export class TrainingStorageStack extends cdk.Stack {
  public readonly checkpointBucket: s3.Bucket;
  public readonly trainingRepository: ecr.Repository;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    this.checkpointBucket = new s3.Bucket(this, 'CheckpointBucket', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      versioned: true,
      lifecycleRules: [
        {
          abortIncompleteMultipartUploadAfter: cdk.Duration.days(7),
          noncurrentVersionExpiration: cdk.Duration.days(30),
        },
      ],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      autoDeleteObjects: false,
    });
    const cfnCheckpointBucket = this.checkpointBucket.node.defaultChild as s3.CfnBucket;
    cfnCheckpointBucket.cfnOptions.metadata = {
      checkov: {
        skip: [
          {
            id: 'CKV_AWS_18',
            comment: 'Dev training artifacts are private and versioned; a persistent access-log bucket is not warranted.',
          },
        ],
      },
    };

    this.trainingRepository = new ecr.Repository(this, 'TrainingRepository', {
      repositoryName: 'generals-training',
      encryption: ecr.RepositoryEncryption.AES_256,
      imageScanOnPush: true,
      imageTagMutability: ecr.TagMutability.IMMUTABLE,
      lifecycleRules: [{ maxImageCount: 30 }],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      emptyOnDelete: false,
    });
    const cfnTrainingRepository = this.trainingRepository.node.defaultChild as ecr.CfnRepository;
    cfnTrainingRepository.cfnOptions.metadata = {
      checkov: {
        skip: [
          {
            id: 'CKV_AWS_136',
            comment: 'ECR AES-256 service-managed encryption is sufficient for non-sensitive training images.',
          },
        ],
      },
    };

    new cdk.CfnOutput(this, 'CheckpointBucketName', {
      value: this.checkpointBucket.bucketName,
      description: 'Durable destination for PPO checkpoints and logs',
    });
    new cdk.CfnOutput(this, 'RepositoryUri', {
      value: this.trainingRepository.repositoryUri,
      description: 'ECR repository for immutable Generals training images',
    });
  }
}

export interface TrainingNetworkExtensionStackProps extends cdk.StackProps {
  readonly vpcId: string;
  readonly publicRouteTableId: string;
}

export class TrainingNetworkExtensionStack extends cdk.Stack {
  public readonly capacityAvailabilityZones = [
    'us-east-1c',
    'us-east-1d',
    'us-east-1e',
    'us-east-1f',
  ];
  public readonly publicSubnetIds: string[];

  constructor(scope: Construct, id: string, props: TrainingNetworkExtensionStackProps) {
    super(scope, id, props);

    cdk.Validations.of(this).acknowledge({
      id: 'CloudFormation-Validate::W3010',
      reason: 'Exact fallback zones are intentional because Batch observed p5.4xlarge capacity failures in 1a/1b.',
    });
    this.publicSubnetIds = this.capacityAvailabilityZones.map((availabilityZone, index) => {
      const subnet = new ec2.CfnSubnet(this, `PublicSubnet${index + 3}`, {
        availabilityZone,
        cidrBlock: `10.40.${index + 3}.0/24`,
        mapPublicIpOnLaunch: true,
        vpcId: props.vpcId,
        tags: [
          { key: 'Name', value: `ml-lab-public-${availabilityZone}` },
          { key: 'Network', value: 'Public' },
        ],
      });
      new ec2.CfnSubnetRouteTableAssociation(this, `PublicRouteAssociation${index + 3}`, {
        routeTableId: props.publicRouteTableId,
        subnetId: subnet.ref,
      });
      return subnet.ref;
    });

    new cdk.CfnOutput(this, 'AdditionalPublicSubnetIds', {
      value: cdk.Fn.join(',', this.publicSubnetIds),
      description: 'Public subnets added for H100 capacity fallback',
    });
  }
}

export interface TrainingBatchStackProps extends cdk.StackProps {
  readonly checkpointBucket: s3.IBucket;
  readonly trainingRepository: ecr.IRepository;
  readonly maxvCpus?: number;
  readonly spotMaxvCpus?: number;
  readonly highMemoryMaxvCpus?: number;
  readonly highMemorySpotMaxvCpus?: number;
  readonly vpc?: ec2.IVpc;
  readonly securityGroup?: ec2.ISecurityGroup;
  readonly vpcId?: string;
  readonly availabilityZones?: string[];
  readonly publicSubnetIds?: string[];
  readonly publicSubnetRouteTableIds?: string[];
  readonly securityGroupId?: string;
  readonly submitterRoleName?: string;
  /** Override when a region does not offer every default high-memory type (e.g. no p4de in us-east-2). */
  readonly highMemoryInstanceTypes?: string[];
}

export class TrainingBatchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: TrainingBatchStackProps) {
    super(scope, id, props);

    if (!props.vpc && (!props.vpcId || !props.availabilityZones || !props.publicSubnetIds)) {
      throw new Error('Imported GPU VPC id, availability zones, and public subnet ids are required');
    }
    if (!props.securityGroup && !props.securityGroupId) {
      throw new Error('Imported GPU security group id is required');
    }
    const imageTag = new cdk.CfnParameter(this, 'ImageTag', {
      type: 'String',
      description: 'Existing immutable ECR image tag for this Batch job definition',
      allowedPattern: '[A-Za-z0-9_][A-Za-z0-9_.-]{0,299}',
      constraintDescription: 'must be a valid, non-empty ECR image tag',
    });
    const vpc = props.vpc ?? ec2.Vpc.fromVpcAttributes(this, 'GpuVpc', {
      vpcId: props.vpcId!,
      availabilityZones: props.availabilityZones!,
      publicSubnetIds: props.publicSubnetIds!,
      publicSubnetRouteTableIds: props.publicSubnetRouteTableIds,
    });
    const securityGroup = props.securityGroup ?? ec2.SecurityGroup.fromSecurityGroupId(
      this,
      'GpuSecurityGroup',
      props.securityGroupId!,
      { mutable: false },
    );

    const logGroup = new logs.LogGroup(this, 'TrainingLogs', {
      logGroupName: '/aws/batch/generals-training',
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });
    const cfnLogGroup = logGroup.node.defaultChild as logs.CfnLogGroup;
    cfnLogGroup.cfnOptions.metadata = {
      checkov: {
        skip: [
          {
            id: 'CKV_AWS_158',
            comment: 'CloudWatch service-managed encryption is sufficient for non-sensitive training stdout.',
          },
        ],
      },
    };

    const jobRole = new iam.Role(this, 'TrainingJobRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'Read and write Generals experiment artifacts under the project S3 prefix',
    });
    jobRole.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetBucketLocation', 's3:ListBucket'],
      resources: [props.checkpointBucket.bucketArn],
      conditions: {
        StringLike: {
          's3:prefix': ['generals/experiments/*'],
        },
      },
    }));
    jobRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        's3:GetObject',
        's3:PutObject',
        's3:AbortMultipartUpload',
        's3:ListMultipartUploadParts',
      ],
      resources: [`${props.checkpointBucket.bucketArn}/generals/experiments/*`],
    }));

    const launchTemplate = new ec2.LaunchTemplate(this, 'GpuLaunchTemplate', {
      requireImdsv2: true,
      blockDevices: [{
        deviceName: '/dev/xvda',
        volume: ec2.BlockDeviceVolume.ebs(100, {
          encrypted: true,
          deleteOnTermination: true,
          volumeType: ec2.EbsDeviceVolumeType.GP3,
        }),
      }],
    });

    const computeEnvironment = new batch.ManagedEc2EcsComputeEnvironment(this, 'GpuOnDemand', {
      computeEnvironmentName: 'generals-gpu-ondemand',
      vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PUBLIC,
        onePerAz: true,
      },
      securityGroups: [securityGroup],
      allocationStrategy: batch.AllocationStrategy.BEST_FIT_PROGRESSIVE,
      instanceTypes: [
        new ec2.InstanceType('g6e.xlarge'),
        new ec2.InstanceType('g6.xlarge'),
        new ec2.InstanceType('g5.xlarge'),
      ],
      useOptimalInstanceClasses: false,
      images: [{ imageType: batch.EcsMachineImageType.ECS_AL2023_NVIDIA }],
      launchTemplate,
      minvCpus: 0,
      maxvCpus: props.maxvCpus ?? 4,
      spot: false,
      replaceComputeEnvironment: false,
    });

    const jobQueue = new batch.JobQueue(this, 'GpuTrainingQueue', {
      jobQueueName: 'gpu-training',
      priority: 1,
      computeEnvironments: [{ computeEnvironment, order: 1 }],
    });

    const spotComputeEnvironment = new batch.ManagedEc2EcsComputeEnvironment(this, 'GpuSpot', {
      computeEnvironmentName: 'generals-gpu-spot',
      vpc,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PUBLIC,
        onePerAz: true,
      },
      securityGroups: [securityGroup],
      allocationStrategy: batch.AllocationStrategy.SPOT_PRICE_CAPACITY_OPTIMIZED,
      instanceTypes: [
        new ec2.InstanceType('g6e.xlarge'),
        new ec2.InstanceType('g6.xlarge'),
        new ec2.InstanceType('g5.xlarge'),
      ],
      useOptimalInstanceClasses: false,
      images: [{ imageType: batch.EcsMachineImageType.ECS_AL2023_NVIDIA }],
      launchTemplate,
      minvCpus: 0,
      maxvCpus: props.spotMaxvCpus ?? 4,
      spot: true,
      replaceComputeEnvironment: false,
    });

    const spotJobQueue = new batch.JobQueue(this, 'GpuSpotTrainingQueue', {
      jobQueueName: 'gpu-training-spot',
      priority: 1,
      computeEnvironments: [{ computeEnvironment: spotComputeEnvironment, order: 1 }],
    });

    const highMemoryComputeEnvironment = new batch.ManagedEc2EcsComputeEnvironment(
      this,
      'HighMemoryGpuOnDemand',
      {
        computeEnvironmentName: 'generals-high-memory-gpu-ondemand',
        vpc,
        vpcSubnets: {
          subnetType: ec2.SubnetType.PUBLIC,
          onePerAz: true,
        },
        securityGroups: [securityGroup],
        allocationStrategy: batch.AllocationStrategy.BEST_FIT_PROGRESSIVE,
        instanceTypes: (
          props.highMemoryInstanceTypes ?? ['g7e.2xlarge', 'p5.4xlarge', 'p4de.24xlarge']
        ).map((name) => new ec2.InstanceType(name)),
        useOptimalInstanceClasses: false,
        images: [{ imageType: batch.EcsMachineImageType.ECS_AL2023_NVIDIA }],
        launchTemplate,
        minvCpus: 0,
        maxvCpus: props.highMemoryMaxvCpus ?? 96,
        updateToLatestImageVersion: true,
        spot: false,
        replaceComputeEnvironment: false,
      },
    );

    const highMemoryJobQueue = new batch.JobQueue(this, 'HighMemoryGpuTrainingQueue', {
      jobQueueName: 'gpu-training-high-memory',
      priority: 1,
      computeEnvironments: [{ computeEnvironment: highMemoryComputeEnvironment, order: 1 }],
    });

    const highMemorySpotComputeEnvironment = new batch.ManagedEc2EcsComputeEnvironment(
      this,
      'HighMemoryGpuSpot',
      {
        computeEnvironmentName: 'generals-high-memory-gpu-spot',
        vpc,
        vpcSubnets: {
          subnetType: ec2.SubnetType.PUBLIC,
          onePerAz: true,
        },
        securityGroups: [securityGroup],
        allocationStrategy: batch.AllocationStrategy.SPOT_PRICE_CAPACITY_OPTIMIZED,
        instanceTypes: (
          props.highMemoryInstanceTypes ?? ['p5.4xlarge', 'p4de.24xlarge']
        ).filter((name) => name !== 'g7e.2xlarge')
          .map((name) => new ec2.InstanceType(name)),
        useOptimalInstanceClasses: false,
        images: [{ imageType: batch.EcsMachineImageType.ECS_AL2023_NVIDIA }],
        launchTemplate,
        minvCpus: 0,
        maxvCpus: props.highMemorySpotMaxvCpus ?? 96,
        updateToLatestImageVersion: true,
        spot: true,
        replaceComputeEnvironment: false,
      },
    );

    const highMemorySpotJobQueue = new batch.JobQueue(this, 'HighMemoryGpuSpotTrainingQueue', {
      jobQueueName: 'gpu-training-high-memory-spot',
      priority: 1,
      computeEnvironments: [{ computeEnvironment: highMemorySpotComputeEnvironment, order: 1 }],
    });

    const container = new batch.EcsEc2ContainerDefinition(this, 'TrainingContainer', {
      image: ecs.ContainerImage.fromEcrRepository(
        props.trainingRepository,
        imageTag.valueAsString,
      ),
      command: ['python', 'jobs/runtime.py', '--', 'python', 'jobs/gpu_smoke.py'],
      cpu: 4,
      memory: cdk.Size.mebibytes(15000),
      gpu: 1,
      jobRole,
      environment: {
        PYTHONUNBUFFERED: '1',
        JAX_PLATFORMS: 'cuda',
        ARTIFACT_SYNC_INTERVAL_SECONDS: '300',
      },
      logging: ecs.LogDrivers.awsLogs({
        logGroup,
        streamPrefix: 'training',
      }),
    });

    const jobDefinition = new batch.EcsJobDefinition(this, 'TrainingJobDefinition', {
      jobDefinitionName: 'generals-training',
      container,
      retryAttempts: 3,
      retryStrategies: [
        batch.RetryStrategy.of(
          batch.Action.RETRY,
          batch.Reason.custom({ onExitCode: '75' }),
        ),
        batch.RetryStrategy.of(
          batch.Action.RETRY,
          batch.Reason.custom({ onStatusReason: 'Host EC2*' }),
        ),
        batch.RetryStrategy.of(
          batch.Action.RETRY,
          batch.Reason.custom({ onReason: 'CannotInspectContainerError*' }),
        ),
        batch.RetryStrategy.of(
          batch.Action.RETRY,
          batch.Reason.custom({ onReason: 'DockerTimeoutError*' }),
        ),
        batch.RetryStrategy.of(
          batch.Action.EXIT,
          batch.Reason.custom({ onReason: '*' }),
        ),
      ],
      timeout: cdk.Duration.days(7),
      propagateTags: true,
    });

    const submitterRole = props.submitterRoleName
      ? iam.Role.fromRoleName(this, 'JobSubmitterRole', props.submitterRoleName, { mutable: true })
      : new iam.Role(this, 'JobSubmitterRole', {
        roleName: 'generals-training-job-submitter',
        description: 'Least-privilege role for submitting and observing Generals AWS Batch jobs',
        assumedBy: new iam.AccountPrincipal(this.account).withConditions({
          ArnLike: {
            'aws:PrincipalArn': this.formatArn({
              service: 'iam',
              region: '',
              resource: 'role',
              resourceName: 'aws-reserved/sso.amazonaws.com/AWSReservedSSO_AdministratorAccess_*',
            }),
          },
        }),
      });
    submitterRole.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['batch:SubmitJob'],
      resources: [
        jobQueue.jobQueueArn,
        spotJobQueue.jobQueueArn,
        highMemoryJobQueue.jobQueueArn,
        highMemorySpotJobQueue.jobQueueArn,
        jobDefinition.jobDefinitionArn,
      ],
    }));
    submitterRole.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['batch:TagResource'],
      resources: [
        jobQueue.jobQueueArn,
        spotJobQueue.jobQueueArn,
        highMemoryJobQueue.jobQueueArn,
        highMemorySpotJobQueue.jobQueueArn,
        jobDefinition.jobDefinitionArn,
      ],
    }));
    submitterRole.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['batch:DescribeJobs', 'batch:ListJobs'],
      resources: ['*'],
    }));
    submitterRole.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['batch:CancelJob', 'batch:TerminateJob'],
      resources: [this.formatArn({
        service: 'batch',
        resource: 'job',
        resourceName: '*',
        arnFormat: cdk.ArnFormat.SLASH_RESOURCE_NAME,
      })],
    }));
    submitterRole.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['cloudformation:DescribeStacks'],
      resources: [this.stackId],
    }));
    submitterRole.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: [
        'logs:DescribeLogStreams',
        'logs:GetLogEvents',
        'logs:FilterLogEvents',
      ],
      resources: [`${logGroup.logGroupArn}:*`],
    }));
    submitterRole.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: ['s3:GetBucketLocation', 's3:ListBucket'],
      resources: [props.checkpointBucket.bucketArn],
      conditions: {
        StringLike: {
          's3:prefix': ['generals/experiments/*'],
        },
      },
    }));
    submitterRole.addToPrincipalPolicy(new iam.PolicyStatement({
      actions: [
        's3:GetObject',
        's3:PutObject',
        's3:AbortMultipartUpload',
        's3:ListMultipartUploadParts',
      ],
      resources: [`${props.checkpointBucket.bucketArn}/generals/experiments/*`],
    }));
    props.trainingRepository.grantPullPush(submitterRole);

    new cdk.CfnOutput(this, 'ArtifactBucketName', {
      value: props.checkpointBucket.bucketName,
    });
    new cdk.CfnOutput(this, 'ComputeEnvironmentName', {
      value: computeEnvironment.computeEnvironmentName,
    });
    new cdk.CfnOutput(this, 'JobQueueArn', {
      value: jobQueue.jobQueueArn,
    });
    new cdk.CfnOutput(this, 'JobQueueName', {
      value: jobQueue.jobQueueName,
    });
    new cdk.CfnOutput(this, 'SpotComputeEnvironmentName', {
      value: spotComputeEnvironment.computeEnvironmentName,
    });
    new cdk.CfnOutput(this, 'SpotJobQueueArn', {
      value: spotJobQueue.jobQueueArn,
    });
    new cdk.CfnOutput(this, 'SpotJobQueueName', {
      value: spotJobQueue.jobQueueName,
    });
    new cdk.CfnOutput(this, 'HighMemoryComputeEnvironmentName', {
      value: highMemoryComputeEnvironment.computeEnvironmentName,
    });
    new cdk.CfnOutput(this, 'HighMemoryJobQueueArn', {
      value: highMemoryJobQueue.jobQueueArn,
    });
    new cdk.CfnOutput(this, 'HighMemoryJobQueueName', {
      value: highMemoryJobQueue.jobQueueName,
    });
    new cdk.CfnOutput(this, 'HighMemorySpotComputeEnvironmentName', {
      value: highMemorySpotComputeEnvironment.computeEnvironmentName,
    });
    new cdk.CfnOutput(this, 'HighMemorySpotJobQueueArn', {
      value: highMemorySpotJobQueue.jobQueueArn,
    });
    new cdk.CfnOutput(this, 'HighMemorySpotJobQueueName', {
      value: highMemorySpotJobQueue.jobQueueName,
    });
    new cdk.CfnOutput(this, 'JobDefinitionArn', {
      value: jobDefinition.jobDefinitionArn,
    });
    new cdk.CfnOutput(this, 'JobDefinitionName', {
      value: jobDefinition.jobDefinitionName,
    });
    new cdk.CfnOutput(this, 'RepositoryUri', {
      value: props.trainingRepository.repositoryUri,
    });
    const imageTagOutput = new cdk.CfnOutput(this, 'ImageTagOutput', {
      value: imageTag.valueAsString,
    });
    imageTagOutput.overrideLogicalId('ImageTag');
    new cdk.CfnOutput(this, 'LogGroupName', {
      value: logGroup.logGroupName,
    });
    new cdk.CfnOutput(this, 'MaxVCpus', {
      value: String(props.maxvCpus ?? 4),
    });
    new cdk.CfnOutput(this, 'SpotMaxVCpus', {
      value: String(props.spotMaxvCpus ?? 4),
    });
    new cdk.CfnOutput(this, 'HighMemoryMaxVCpus', {
      value: String(props.highMemoryMaxvCpus ?? 96),
    });
    new cdk.CfnOutput(this, 'HighMemorySpotMaxVCpus', {
      value: String(props.highMemorySpotMaxvCpus ?? 96),
    });
    new cdk.CfnOutput(this, 'JobSubmitterRoleArn', {
      value: submitterRole.roleArn,
    });
  }
}
