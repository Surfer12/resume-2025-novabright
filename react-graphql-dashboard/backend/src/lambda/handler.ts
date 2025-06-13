import { APIGatewayProxyEvent, APIGatewayProxyResult, Context } from 'aws-lambda';
import { ApolloServer } from '@apollo/server';
import { startServerAndCreateLambdaHandler } from '@apollo/server-integrations/aws-lambda';
import { ApolloServerPluginCacheControl } from '@apollo/server/plugin/cacheControl';
import { ApolloServerPluginResponseCache } from '@apollo/server/plugin/responseCache';
import { ApolloServerPluginLandingPageDisabled } from '@apollo/server/plugin/disabled';
import { typeDefs } from '../schema/typedefs';
import { resolvers, createContext } from '../resolvers';
import { initializeServices } from '../services/init';
import { logger } from '../utils/logger';

// Global variables for connection reuse (Lambda container reuse optimization)
let server: ApolloServer;
let isInitialized = false;

// Performance monitoring metrics
const performanceMetrics = {
  coldStarts: 0,
  warmStarts: 0,
  totalRequests: 0,
  averageLatency: 0,
  errors: 0,
};

// Initialize server once per container (outside handler for reuse)
const initializeServer = async () => {
  if (isInitialized) return server;

  const startTime = Date.now();
  
  try {
    // Initialize external services with connection pooling
    await initializeServices();
    
    // Create Apollo Server with optimizations
    server = new ApolloServer({
      typeDefs,
      resolvers,
      plugins: [
        // Disable GraphQL Playground in production
        ApolloServerPluginLandingPageDisabled(),
        
        // Cache control for HTTP caching
        ApolloServerPluginCacheControl({
          defaultMaxAge: 300, // 5 minutes default
          calculateHttpCacheControlHeader: true,
        }),
        
        // Response caching plugin for GraphQL query caching
        ApolloServerPluginResponseCache({
          ttl: 300000, // 5 minutes in milliseconds
          sessionId: (requestContext) => {
            // Cache by user ID for personalized data
            const userId = requestContext.request.http?.headers.get('x-user-id');
            return userId || 'anonymous';
          },
        }),
        
        // Performance monitoring plugin
        {
          async requestDidStart() {
            const requestStartTime = Date.now();
            
            return {
              async didResolveOperation(requestContext) {
                logger.info(`GraphQL operation: ${requestContext.request.operationName}`);
              },
              
              async willSendResponse(requestContext) {
                const duration = Date.now() - requestStartTime;
                
                // Update performance metrics
                performanceMetrics.totalRequests++;
                performanceMetrics.averageLatency = 
                  (performanceMetrics.averageLatency * (performanceMetrics.totalRequests - 1) + duration) 
                  / performanceMetrics.totalRequests;
                
                // Log slow queries
                if (duration > 1000) {
                  logger.warn(`Slow GraphQL query: ${requestContext.request.operationName} took ${duration}ms`);
                }
                
                // Add performance headers
                requestContext.response.http.headers.set('x-response-time', duration.toString());
                requestContext.response.http.headers.set('x-cache-status', 
                  requestContext.response.http.headers.get('cache-control') ? 'HIT' : 'MISS');
              },
              
              async didEncounterErrors(requestContext) {
                performanceMetrics.errors++;
                logger.error('GraphQL errors:', requestContext.errors);
              },
            };
          },
        },
      ],
      
      // Enable introspection only in development
      introspection: process.env.NODE_ENV !== 'production',
      
      // Format errors for production
      formatError: (err) => {
        // Log full error details
        logger.error('GraphQL error:', err);
        
        // Return sanitized error in production
        if (process.env.NODE_ENV === 'production') {
          return new Error('Internal server error');
        }
        
        return err;
      },
    });

    await server.start();
    
    const initTime = Date.now() - startTime;
    logger.info(`Apollo Server initialized in ${initTime}ms`);
    
    isInitialized = true;
    performanceMetrics.coldStarts++;
    
    return server;
  } catch (error) {
    logger.error('Failed to initialize Apollo Server:', error);
    throw error;
  }
};

// Create Lambda handler with optimizations
const createHandler = async () => {
  const server = await initializeServer();
  
  return startServerAndCreateLambdaHandler(
    server,
    {
      // Custom context creation with request optimization
      context: async ({ event, context: lambdaContext }) => {
        const requestContext = await createContext({
          headers: event.headers,
          user: event.requestContext.authorizer?.user,
        });
        
        return {
          ...requestContext,
          lambdaEvent: event,
          lambdaContext,
          requestId: event.requestContext.requestId,
        };
      },
      
      // Middleware for request preprocessing
      middleware: [
        // CORS middleware
        (handler) => async (event, context) => {
          const response = await handler(event, context);
          
          return {
            ...response,
            headers: {
              ...response.headers,
              'Access-Control-Allow-Origin': process.env.CORS_ORIGIN || '*',
              'Access-Control-Allow-Headers': 'Content-Type,Authorization,X-User-ID,X-Request-ID',
              'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
              'Access-Control-Max-Age': '86400',
            },
          };
        },
        
        // Compression middleware
        (handler) => async (event, context) => {
          const response = await handler(event, context);
          
          // Add compression headers for large responses
          if (response.body && response.body.length > 1024) {
            response.headers = {
              ...response.headers,
              'Content-Encoding': 'gzip',
            };
          }
          
          return response;
        },
      ],
    }
  );
};

// Main Lambda handler with performance optimizations
export const graphqlHandler = async (
  event: APIGatewayProxyEvent,
  context: Context
): Promise<APIGatewayProxyResult> => {
  const requestStartTime = Date.now();
  
  // Optimize Lambda execution
  context.callbackWaitsForEmptyEventLoop = false;
  
  try {
    // Handle preflight requests
    if (event.httpMethod === 'OPTIONS') {
      return {
        statusCode: 200,
        headers: {
          'Access-Control-Allow-Origin': process.env.CORS_ORIGIN || '*',
          'Access-Control-Allow-Headers': 'Content-Type,Authorization,X-User-ID,X-Request-ID',
          'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
          'Access-Control-Max-Age': '86400',
        },
        body: '',
      };
    }
    
    // Track warm vs cold starts
    if (isInitialized) {
      performanceMetrics.warmStarts++;
    }
    
    // Get or create handler
    const handler = await createHandler();
    
    // Execute GraphQL request
    const response = await handler(event, context);
    
    const duration = Date.now() - requestStartTime;
    
    // Add performance metrics to response headers
    const enhancedResponse = {
      ...response,
      headers: {
        ...response.headers,
        'x-lambda-duration': duration.toString(),
        'x-cold-start': isInitialized ? 'false' : 'true',
        'x-performance-stats': JSON.stringify({
          coldStarts: performanceMetrics.coldStarts,
          warmStarts: performanceMetrics.warmStarts,
          totalRequests: performanceMetrics.totalRequests,
          averageLatency: Math.round(performanceMetrics.averageLatency),
          errorRate: (performanceMetrics.errors / performanceMetrics.totalRequests * 100).toFixed(2),
        }),
      },
    };
    
    // Log performance metrics
    logger.info(`Lambda execution completed in ${duration}ms`, {
      requestId: event.requestContext.requestId,
      coldStart: !isInitialized,
      duration,
      method: event.httpMethod,
      path: event.path,
    });
    
    return enhancedResponse;
    
  } catch (error) {
    logger.error('Lambda handler error:', error);
    
    const duration = Date.now() - requestStartTime;
    performanceMetrics.errors++;
    
    return {
      statusCode: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': process.env.CORS_ORIGIN || '*',
        'x-lambda-duration': duration.toString(),
        'x-error': 'true',
      },
      body: JSON.stringify({
        error: process.env.NODE_ENV === 'production' 
          ? 'Internal server error' 
          : error.message,
        requestId: event.requestContext.requestId,
      }),
    };
  }
};

// Health check handler for load balancer
export const healthCheckHandler = async (
  event: APIGatewayProxyEvent
): Promise<APIGatewayProxyResult> => {
  return {
    statusCode: 200,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
    body: JSON.stringify({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: process.env.VERSION || '1.0.0',
      metrics: performanceMetrics,
      uptime: process.uptime(),
    }),
  };
};

// Warmup handler to prevent cold starts
export const warmupHandler = async (): Promise<APIGatewayProxyResult> => {
  if (!isInitialized) {
    await initializeServer();
  }
  
  return {
    statusCode: 200,
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: 'Lambda warmed up successfully',
      initialized: isInitialized,
      timestamp: new Date().toISOString(),
    }),
  };
};