import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as cdk from 'aws-cdk-lib/core';
import { Construct } from 'constructs';

export interface TrainingRegionalFoundationStackProps extends cdk.StackProps {
  readonly artifactBucketName: string;
  /** Unique per fallback region so VPCs could be peered later if needed. */
  readonly vpcCidr?: string;
}

/**
 * Regional resources that cannot be shared with the primary us-east-1 stack.
 *
 * The artifact bucket remains in us-east-1 so experiment IDs and output paths
 * are common across regions. Batch, ECR, and networking are regional.
 */
export class TrainingRegionalFoundationStack extends cdk.Stack {
  public readonly artifactBucket: s3.IBucket;
  public readonly trainingRepository: ecr.Repository;
  public readonly vpc: ec2.Vpc;
  public readonly securityGroup: ec2.SecurityGroup;

  constructor(
    scope: Construct,
    id: string,
    props: TrainingRegionalFoundationStackProps,
  ) {
    super(scope, id, props);

    cdk.Validations.of(this).acknowledge({
      id: 'CloudFormation-Validate::W3010',
      reason: 'The fallback VPC intentionally spans the fallback region Availability Zones.',
    });

    this.artifactBucket = s3.Bucket.fromBucketName(
      this,
      'PrimaryArtifactBucket',
      props.artifactBucketName,
    );

    this.vpc = new ec2.Vpc(this, 'TrainingVpc', {
      ipAddresses: ec2.IpAddresses.cidr(props.vpcCidr ?? '10.41.0.0/16'),
      maxAzs: 4,
      natGateways: 0,
      subnetConfiguration: [{
        cidrMask: 24,
        name: 'public',
        subnetType: ec2.SubnetType.PUBLIC,
      }],
    });
    const restrictDefaultSgProvider = this.node.tryFindChild(
      'Custom::VpcRestrictDefaultSGCustomResourceProvider',
    );
    const restrictDefaultSgHandlerConstruct = restrictDefaultSgProvider?.node
      .tryFindChild('Handler');
    const restrictDefaultSgHandler = (
      restrictDefaultSgHandlerConstruct instanceof cdk.CfnResource
        ? restrictDefaultSgHandlerConstruct
        : restrictDefaultSgHandlerConstruct?.node.defaultChild
    ) as cdk.CfnResource | undefined;
    if (restrictDefaultSgHandler) {
      restrictDefaultSgHandler.cfnOptions.metadata = {
        checkov: {
          skip: [
            {
              id: 'CKV_AWS_115',
              comment: 'CDK-owned singleton runs only during stack changes; reserved concurrency is unnecessary.',
            },
            {
              id: 'CKV_AWS_116',
              comment: 'CloudFormation reports custom-resource failures directly; a DLQ would not recover them.',
            },
            {
              id: 'CKV_AWS_117',
              comment: 'The default-security-group cleanup provider must remain outside the VPC it configures.',
            },
          ],
        },
      };
    }

    this.securityGroup = new ec2.SecurityGroup(this, 'GpuSecurityGroup', {
      vpc: this.vpc,
      allowAllOutbound: true,
      description: 'No-ingress security group for regional AWS Batch GPU instances',
    });

    this.trainingRepository = new ecr.Repository(this, 'TrainingRepository', {
      repositoryName: 'generals-training',
      encryption: ecr.RepositoryEncryption.AES_256,
      imageScanOnPush: true,
      imageTagMutability: ecr.TagMutability.IMMUTABLE,
      lifecycleRules: [{ maxImageCount: 30 }],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      emptyOnDelete: false,
    });
    const cfnRepository = this.trainingRepository.node.defaultChild as ecr.CfnRepository;
    cfnRepository.cfnOptions.metadata = {
      checkov: {
        skip: [{
          id: 'CKV_AWS_136',
          comment: 'ECR AES-256 service-managed encryption is sufficient for non-sensitive training images.',
        }],
      },
    };

    new cdk.CfnOutput(this, 'ArtifactBucketName', {
      value: this.artifactBucket.bucketName,
    });
    new cdk.CfnOutput(this, 'RepositoryUri', {
      value: this.trainingRepository.repositoryUri,
    });
    new cdk.CfnOutput(this, 'VpcId', {
      value: this.vpc.vpcId,
    });
    new cdk.CfnOutput(this, 'SecurityGroupId', {
      value: this.securityGroup.securityGroupId,
    });
  }
}
