import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigatewayv2';
import * as apigatewayIntegrations from '@aws-cdk/aws-apigatewayv2-integrations-alpha';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import { Construct } from 'constructs';

interface DashboardStackProps extends cdk.StackProps {
  environment: 'development' | 'staging' | 'production';
  domainName?: string;
}

export class DashboardStack extends cdk.Stack {
  public readonly apiUrl: string;
  public readonly distributionUrl: string;

  constructor(scope: Construct, id: string, props: DashboardStackProps) {
    super(scope, id, props);

    const { environment } = props;
    const isProd = environment === 'production';

    // DynamoDB Tables with optimized configuration
    const usersTable = new dynamodb.Table(this, 'UsersTable', {
      tableName: `dashboard-users-${environment}`,
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: isProd,
      encryption: isProd ? dynamodb.TableEncryption.AWS_MANAGED : undefined,
      removalPolicy: isProd ? cdk.RemovalPolicy.RETAIN : cdk.RemovalPolicy.DESTROY,
    });

    // GSI for efficient queries
    usersTable.addGlobalSecondaryIndex({
      indexName: 'EmailIndex',
      partitionKey: { name: 'email', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    usersTable.addGlobalSecondaryIndex({
      indexName: 'RoleIndex',
      partitionKey: { name: 'role', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'createdAt', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    const metricsTable = new dynamodb.Table(this, 'MetricsTable', {
      tableName: `dashboard-metrics-${environment}`,
      partitionKey: { name: 'type', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'ttl',
      pointInTimeRecovery: isProd,
      removalPolicy: isProd ? cdk.RemovalPolicy.RETAIN : cdk.RemovalPolicy.DESTROY,
    });

    const activitiesTable = new dynamodb.Table(this, 'ActivitiesTable', {
      tableName: `dashboard-activities-${environment}`,
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'ttl',
      removalPolicy: isProd ? cdk.RemovalPolicy.RETAIN : cdk.RemovalPolicy.DESTROY,
    });

    // ElastiCache Redis for caching (production only)
    let redisCluster: elasticache.CfnCacheCluster | undefined;
    if (isProd) {
      const redisSubnetGroup = new elasticache.CfnSubnetGroup(this, 'RedisSubnetGroup', {
        description: 'Subnet group for Redis cluster',
        subnetIds: cdk.Fn.importListValue('dashboard-vpc-private-subnet-ids'),
      });

      redisCluster = new elasticache.CfnCacheCluster(this, 'RedisCluster', {
        cacheNodeType: 'cache.r6g.large',
        engine: 'redis',
        numCacheNodes: 1,
        cacheSubnetGroupName: redisSubnetGroup.ref,
        vpcSecurityGroupIds: [
          cdk.Fn.importValue('dashboard-vpc-redis-security-group-id'),
        ],
      });
    }

    // Lambda Layer for shared dependencies
    const dependenciesLayer = new lambda.LayerVersion(this, 'DependenciesLayer', {
      code: lambda.Code.fromAsset('../backend/lambda-layer'),
      compatibleRuntimes: [lambda.Runtime.NODEJS_18_X],
      description: 'Shared dependencies for GraphQL Lambda',
    });

    // Main GraphQL Lambda with optimizations
    const graphqlLambda = new lambda.Function(this, 'GraphQLFunction', {
      functionName: `dashboard-graphql-${environment}`,
      runtime: lambda.Runtime.NODEJS_18_X,
      handler: 'lambda/handler.graphqlHandler',
      code: lambda.Code.fromAsset('../backend/dist'),
      layers: [dependenciesLayer],
      
      // Performance optimizations
      memorySize: isProd ? 1024 : 512,
      timeout: cdk.Duration.seconds(30),
      reservedConcurrentExecutions: isProd ? 100 : undefined,
      
      // Environment variables
      environment: {
        NODE_ENV: environment,
        USERS_TABLE: usersTable.tableName,
        METRICS_TABLE: metricsTable.tableName,
        ACTIVITIES_TABLE: activitiesTable.tableName,
        REDIS_ENDPOINT: redisCluster?.attrRedisEndpointAddress || '',
        CORS_ORIGIN: props.domainName ? `https://${props.domainName}` : '*',
        LOG_LEVEL: isProd ? 'info' : 'debug',
      },
      
      // VPC configuration for Redis access (production only)
      ...(isProd && {
        vpc: cdk.Fn.importValue('dashboard-vpc'),
        vpcSubnets: {
          subnets: cdk.Fn.importListValue('dashboard-vpc-private-subnet-ids').map(
            (subnetId, index) => lambda.Subnet.fromSubnetId(this, `Subnet${index}`, subnetId)
          ),
        },
        securityGroups: [
          iam.SecurityGroup.fromSecurityGroupId(
            this,
            'LambdaSecurityGroup',
            cdk.Fn.importValue('dashboard-vpc-lambda-security-group-id')
          ),
        ],
      }),
      
      // CloudWatch Logs
      logRetention: logs.RetentionDays.ONE_WEEK,
    });

    // Warmup Lambda to prevent cold starts
    const warmupLambda = new lambda.Function(this, 'WarmupFunction', {
      functionName: `dashboard-warmup-${environment}`,
      runtime: lambda.Runtime.NODEJS_18_X,
      handler: 'lambda/handler.warmupHandler',
      code: lambda.Code.fromAsset('../backend/dist'),
      layers: [dependenciesLayer],
      memorySize: 256,
      timeout: cdk.Duration.seconds(10),
      environment: {
        MAIN_FUNCTION_NAME: graphqlLambda.functionName,
      },
    });

    // Grant permissions to Lambda functions
    usersTable.grantReadWriteData(graphqlLambda);
    metricsTable.grantReadWriteData(graphqlLambda);
    activitiesTable.grantReadWriteData(graphqlLambda);
    
    graphqlLambda.addToRolePolicy(new iam.PolicyStatement({
      actions: ['lambda:InvokeFunction'],
      resources: [warmupLambda.functionArn],
    }));

    // API Gateway with optimization
    const api = new apigateway.HttpApi(this, 'DashboardAPI', {
      apiName: `dashboard-api-${environment}`,
      description: 'Optimized GraphQL API for dashboard',
      corsPreflight: {
        allowOrigins: props.domainName ? [`https://${props.domainName}`] : ['*'],
        allowHeaders: ['Content-Type', 'Authorization', 'X-User-ID', 'X-Request-ID'],
        allowMethods: [
          apigateway.CorsHttpMethod.GET,
          apigateway.CorsHttpMethod.POST,
          apigateway.CorsHttpMethod.OPTIONS,
        ],
        maxAge: cdk.Duration.days(1),
      },
    });

    // Lambda integration with caching
    const lambdaIntegration = new apigatewayIntegrations.HttpLambdaIntegration(
      'GraphQLIntegration',
      graphqlLambda,
      {
        payloadFormatVersion: apigateway.PayloadFormatVersion.VERSION_2_0,
      }
    );

    // Routes
    api.addRoute({
      path: '/graphql',
      methods: [apigateway.HttpMethod.POST, apigateway.HttpMethod.GET],
      integration: lambdaIntegration,
    });

    api.addRoute({
      path: '/health',
      methods: [apigateway.HttpMethod.GET],
      integration: new apigatewayIntegrations.HttpLambdaIntegration(
        'HealthIntegration',
        new lambda.Function(this, 'HealthFunction', {
          runtime: lambda.Runtime.NODEJS_18_X,
          handler: 'lambda/handler.healthCheckHandler',
          code: lambda.Code.fromAsset('../backend/dist'),
          memorySize: 128,
          timeout: cdk.Duration.seconds(5),
        })
      ),
    });

    // S3 bucket for frontend hosting
    const websiteBucket = new s3.Bucket(this, 'WebsiteBucket', {
      bucketName: `dashboard-frontend-${environment}-${this.account}`,
      publicReadAccess: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: isProd ? cdk.RemovalPolicy.RETAIN : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: !isProd,
    });

    // CloudFront Origin Access Identity
    const oai = new cloudfront.OriginAccessIdentity(this, 'OAI', {
      comment: `OAI for dashboard-${environment}`,
    });

    websiteBucket.grantRead(oai);

    // CloudFront distribution with optimization
    const distribution = new cloudfront.Distribution(this, 'Distribution', {
      comment: `Dashboard distribution - ${environment}`,
      defaultRootObject: 'index.html',
      
      defaultBehavior: {
        origin: new origins.S3Origin(websiteBucket, { originAccessIdentity: oai }),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
        compress: true,
        allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
      },
      
      additionalBehaviors: {
        '/graphql': {
          origin: new origins.HttpOrigin(cdk.Fn.select(2, cdk.Fn.split('/', api.url!))),
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          cachePolicy: new cloudfront.CachePolicy(this, 'GraphQLCachePolicy', {
            cachePolicyName: `graphql-cache-${environment}`,
            defaultTtl: cdk.Duration.seconds(0), // No caching for GraphQL by default
            maxTtl: cdk.Duration.seconds(300),
            minTtl: cdk.Duration.seconds(0),
            enableAcceptEncodingGzip: true,
            enableAcceptEncodingBrotli: true,
            headerBehavior: cloudfront.CacheHeaderBehavior.allowList(
              'Authorization',
              'Content-Type',
              'X-User-ID',
              'X-Request-ID'
            ),
          }),
          allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
        },
      },
      
      errorResponses: [
        {
          httpStatus: 404,
          responseHttpStatus: 200,
          responsePagePath: '/index.html',
          ttl: cdk.Duration.minutes(5),
        },
      ],
      
      priceClass: isProd 
        ? cloudfront.PriceClass.PRICE_CLASS_ALL 
        : cloudfront.PriceClass.PRICE_CLASS_100,
    });

    // Deploy frontend to S3
    new s3deploy.BucketDeployment(this, 'DeployWebsite', {
      sources: [s3deploy.Source.asset('../frontend/dist')],
      destinationBucket: websiteBucket,
      distribution,
      distributionPaths: ['/*'],
    });

    // CloudWatch Alarms for monitoring
    new cloudwatch.Alarm(this, 'HighLatencyAlarm', {
      metric: graphqlLambda.metricDuration(),
      threshold: 5000, // 5 seconds
      evaluationPeriods: 3,
      alarmDescription: 'GraphQL Lambda high latency',
    });

    new cloudwatch.Alarm(this, 'HighErrorRateAlarm', {
      metric: graphqlLambda.metricErrors(),
      threshold: 10,
      evaluationPeriods: 2,
      alarmDescription: 'GraphQL Lambda high error rate',
    });

    // Scheduled warmup to prevent cold starts
    const warmupRule = new events.Rule(this, 'WarmupRule', {
      schedule: events.Schedule.rate(cdk.Duration.minutes(5)),
      description: 'Keep Lambda warm',
    });

    warmupRule.addTarget(new targets.LambdaFunction(warmupLambda));

    // Outputs
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: api.url!,
      description: 'GraphQL API URL',
    });

    new cdk.CfnOutput(this, 'DistributionUrl', {
      value: `https://${distribution.distributionDomainName}`,
      description: 'CloudFront distribution URL',
    });

    new cdk.CfnOutput(this, 'GraphQLEndpoint', {
      value: `${api.url}graphql`,
      description: 'GraphQL endpoint URL',
    });

    // Store for use in other methods
    this.apiUrl = api.url!;
    this.distributionUrl = `https://${distribution.distributionDomainName}`;
  }
}