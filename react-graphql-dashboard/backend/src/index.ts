import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { ApolloServerPluginDrainHttpServer } from '@apollo/server/plugin/drainHttpServer';
import { ApolloServerPluginCacheControl } from '@apollo/server/plugin/cacheControl';
import { createServer } from 'http';
import express from 'express';
import { WebSocketServer } from 'ws';
import { useServer } from 'graphql-ws/lib/use/ws';
import { makeExecutableSchema } from '@graphql-tools/schema';
import cors from 'cors';
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

// Merge resolvers
const mergedResolvers = {
  Query: {
    ...resolvers.Query,
  },
  Mutation: {
    ...resolvers.Mutation,
  },
  Subscription: {
    ...subscriptionResolvers.Subscription,
  },
};

// Create executable schema
const schema = makeExecutableSchema({
  typeDefs,
  resolvers: mergedResolvers,
});

async function startServer() {
  try {
    // Create Express app
    const app = express();
    
    // Create HTTP server
    const httpServer = createServer(app);
    
    // Create WebSocket server for subscriptions
    const wsServer = new WebSocketServer({
      server: httpServer,
      path: '/graphql',
    });
    
    // Set up WebSocket server for GraphQL subscriptions
    const serverCleanup = useServer(
      {
        schema,
        context: async (ctx: any) => {
          // Create context for WebSocket connections
          return createContext({ req: ctx.extra.request });
        },
      },
      wsServer
    );

    // Initialize Apollo Server with optimizations
    const server = new ApolloServer({
      schema,
      plugins: [
        // Drain HTTP server plugin
        ApolloServerPluginDrainHttpServer({ httpServer }),
        // Clean up WebSocket server
        {
          async serverWillStart() {
            return {
              async drainServer() {
                await serverCleanup.dispose();
              },
            };
          },
        },
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

    // Start Apollo Server
    await server.start();

    // Apply CORS middleware
    app.use(cors({
      origin: ['http://localhost:3000', 'http://localhost:5173'], // Allow both Vite and CRA dev servers
      credentials: true,
    }));

    // Apply Express JSON middleware
    app.use('/graphql', express.json());

    // Apply Apollo GraphQL middleware
    app.use('/graphql', expressMiddleware(server, {
      context: async ({ req }) => {
        return createContext({ req });
      },
    }));

    const PORT = process.env.PORT || 4000;

    // Start HTTP server
    httpServer.listen(PORT, () => {
      logger.info(`🚀 GraphQL Server ready at http://localhost:${PORT}/graphql`);
      logger.info(`🔗 WebSocket subscriptions ready at ws://localhost:${PORT}/graphql`);
      logger.info('🧠 Consciousness visualization endpoints available');
      logger.info('📊 Performance monitoring enabled');
    });
    
    // Start subscription simulation
    startSubscriptionSimulation(metricsService, activityService);
    
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
