import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { ApolloServerPluginCacheControl } from '@apollo/server/plugin/cacheControl';
import { typeDefs } from './schema/typedefs';
import { resolvers, createContext } from './resolvers';
import { logger } from './utils/logger';
import { subscriptionResolvers, startSubscriptionSimulation } from './resolvers/subscriptions';
import { UserService } from './services/UserService';
import { MetricsService } from './services/MetricsService';
import { ActivityService } from './services/ActivityService';

// Initialize services
const userService = new UserService();
const metricsService = new MetricsService();
const activityService = new ActivityService();

// Performance monitoring for development
const performanceMetrics = {
  totalRequests: 0,
  averageLatency: 0,
  errors: 0,
};

// Merge resolvers - temporarily comment out subscriptions
const mergedResolvers = {
  Query: {
    ...resolvers.Query,
  },
  Mutation: {
    ...resolvers.Mutation,
  },
  // TODO: Re-enable subscriptions once WebSocket setup is fixed
  // Subscription: {
  //   ...subscriptionResolvers.Subscription,
  // },
};

async function startServer() {
  try {
    // Initialize Apollo Server with optimizations
    const server = new ApolloServer({
      typeDefs,
      resolvers: mergedResolvers,
      plugins: [
        // Cache control for performance optimization
        ApolloServerPluginCacheControl({
          defaultMaxAge: 300, // 5 minutes default cache
          calculateHttpHeaders: false,
        }),
        // Performance monitoring plugin
        {
          async requestDidStart() {
            const startTime = Date.now();
            return {
              async willSendResponse(requestContext) {
                performanceMetrics.totalRequests++;
                const duration = Date.now() - startTime;
                performanceMetrics.averageLatency = 
                  (performanceMetrics.averageLatency * (performanceMetrics.totalRequests - 1) + duration) / 
                  performanceMetrics.totalRequests;
                
                logger.info('GraphQL Request', {
                  operation: requestContext.request.operationName,
                  duration: `${duration}ms`,
                  totalRequests: performanceMetrics.totalRequests,
                  avgLatency: `${Math.round(performanceMetrics.averageLatency)}ms`
                });
              },
              async didEncounterErrors(requestContext) {
                performanceMetrics.errors++;
                logger.error('GraphQL Error', {
                  errors: requestContext.errors,
                  operation: requestContext.request.operationName
                });
              }
            };
          }
        }
      ],
      // Enable introspection and playground for development
      introspection: true,
    });

    // Start the server
    const { url } = await startStandaloneServer(server, {
      listen: { port: 4000 },
      context: async ({ req }) => {
        // Create context for each request
        return createContext({ req });
      },
    });

    logger.info(`🚀 GraphQL Server ready at ${url}`);
    logger.info('🧠 Consciousness visualization endpoints available');
    logger.info('📊 Performance monitoring enabled');
    logger.warn('⚠️  WebSocket subscriptions temporarily disabled - will be re-enabled once setup is fixed');
    
    // TODO: Re-enable subscription simulation once WebSocket setup is fixed
    // startSubscriptionSimulation(metricsService, activityService);
    
    // Log performance metrics every 30 seconds in development
    setInterval(() => {
      if (performanceMetrics.totalRequests > 0) {
        logger.info('Performance Metrics', {
          totalRequests: performanceMetrics.totalRequests,
          averageLatency: `${Math.round(performanceMetrics.averageLatency)}ms`,
          errors: performanceMetrics.errors,
          errorRate: `${((performanceMetrics.errors / performanceMetrics.totalRequests) * 100).toFixed(2)}%`
        });
      }
    }, 30000);

  } catch (error) {
    logger.error('Failed to start server', error);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  logger.info('Shutting down GraphQL server...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  logger.info('Shutting down GraphQL server...');
  process.exit(0);
});

// Start the server
startServer().catch((error) => {
  logger.error('Server startup failed', error);
  process.exit(1);
});
