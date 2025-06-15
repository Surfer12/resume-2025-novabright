import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { ApolloServerPluginDrainHttpServer } from '@apollo/server/plugin/drainHttpServer';
import { ApolloServerPluginCacheControl } from '@apollo/server/plugin/cacheControl';
import { makeExecutableSchema } from '@graphql-tools/schema';
import { WebSocketServer } from 'ws';
import { useServer } from 'graphql-ws/lib/use/ws';
import express from 'express';
import http from 'http';
import cors from 'cors';
import bodyParser from 'body-parser';
import { typeDefs } from './schema/typedefs';
import { resolvers, createContext } from './resolvers';
import { consciousnessResolvers } from './resolvers/consciousness';
import { logger } from './utils/logger';

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
    ...consciousnessResolvers.Query,
  },
  Mutation: {
    ...resolvers.Mutation,
    ...consciousnessResolvers.Mutation,
  },
  Subscription: {
    ...resolvers.Subscription,
    ...consciousnessResolvers.Subscription,
  },
};

async function startServer() {
  try {
    // Create Express app and HTTP server
    const app = express();
    const httpServer = http.createServer(app);

    // Create WebSocket server for subscriptions
    const wsServer = new WebSocketServer({
      server: httpServer,
      path: '/graphql',
    });

    // Create executable schema
    const schema = makeExecutableSchema({
      typeDefs,
      resolvers: mergedResolvers,
    });

    // Set up WebSocket server for subscriptions
    const serverCleanup = useServer({ 
      schema,
      onConnect: async (ctx) => {
        logger.info('Client connected to WebSocket');
        return true;
      },
      onDisconnect: (ctx, code, reason) => {
        logger.info(`Client disconnected from WebSocket: ${code} - ${reason}`);
      },
      onSubscribe: (ctx, msg) => {
        logger.info(`Subscription started: ${msg.payload.operationName}`);
        return true;
      },
      onError: (ctx, msg, errors) => {
        logger.error('Subscription error', { errors });
      },
    }, wsServer);

    // Initialize Apollo Server with optimizations
    const server = new ApolloServer({
      schema,
      plugins: [
        // Proper HTTP server drain plugin
        ApolloServerPluginDrainHttpServer({ httpServer }),
        // Cleanup for WebSocket server
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

    // Start the Apollo Server
    await server.start();

    // Apply Express middleware
    app.use(
      '/graphql',
      cors<cors.CorsRequest>(),
      bodyParser.json(),
      expressMiddleware(server, {
        context: createContext,
      })
    );

    // Start the HTTP server
    await new Promise<void>((resolve) => httpServer.listen({ port: 4000 }, resolve));

    logger.info(`🚀 GraphQL Server ready at http://localhost:4000/graphql`);
    logger.info(`🔌 WebSocket server ready at ws://localhost:4000/graphql`);
    logger.info('🧠 Consciousness visualization endpoints available');
    logger.info('📊 Performance monitoring enabled');
    
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
