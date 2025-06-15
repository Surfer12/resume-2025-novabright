import {
  ApolloClient,
  InMemoryCache,
  createHttpLink,
  from,
  split,
} from '@apollo/client';
import { setContext } from '@apollo/client/link/context';
import { BatchHttpLink } from '@apollo/client/link/batch-http';
import { createPersistedQueryLink } from '@apollo/client/link/persisted-queries';
import { sha256 } from 'crypto-hash';
import { getMainDefinition } from '@apollo/client/utilities';
import { onError } from '@apollo/client/link/error';

// Environment configuration
const API_ENDPOINT = process.env.VITE_GRAPHQL_ENDPOINT || 'https://api.dashboard.com/graphql';
const WS_ENDPOINT = process.env.VITE_WS_ENDPOINT || 'wss://api.dashboard.com/graphql';

// Performance optimization: Persisted queries for reduced bandwidth
const persistedQueriesLink = createPersistedQueryLink({
  sha256,
  useGETForHashedQueries: false, // Use POST for persisted queries to be compatible with batchHttpLink
});

// Batch HTTP link for query batching (reduces network round trips)
const batchHttpLink = new BatchHttpLink({
  uri: API_ENDPOINT,
  batchMax: 10, // Batch up to 10 queries
  batchInterval: 20, // 20ms batching window
  batchKey: (operation) => {
    // Group operations by priority for optimal batching
    const context = operation.getContext();
    return context.priority || 'default';
  },
});

// Regular HTTP link for non-batchable operations
const httpLink = createHttpLink({
  uri: API_ENDPOINT,
  fetch: (uri, options) => {
    // Add request timing for performance monitoring
    const startTime = performance.now();
    return fetch(uri, options).then(response => {
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      // Log slow queries for optimization
      if (duration > 200) {
        console.warn(`Slow GraphQL query detected: ${duration}ms`, { uri, options });
      }
      
      return response;
    });
  },
});

// Authentication link
const authLink = setContext((_, { headers }) => {
  const token = localStorage.getItem('authToken');
  return {
    headers: {
      ...headers,
      authorization: token ? `Bearer ${token}` : '',
      'x-request-id': crypto.randomUUID(), // For tracing
    },
  };
});

// Error handling link
const errorLink = onError(({ graphQLErrors, networkError, operation, forward }) => {
  if (graphQLErrors) {
    graphQLErrors.forEach(({ message, locations, path }) => {
      console.error(`GraphQL error: ${message}`, { locations, path, operation });
    });
  }

  if (networkError) {
    console.error('Network error:', networkError);
    
    // Retry logic for network errors
    if (networkError && 'statusCode' in networkError && (networkError as any).statusCode === 429) {
      // Rate limited - implement exponential backoff
      return forward(operation);
    }
  }
});

// Split link for batching vs non-batching operations
const splitLink = split(
  ({ query, operationName, getContext }) => {
    const definition = getMainDefinition(query);
    const context = getContext();
    
    // Don't batch mutations or subscriptions
    if (definition.kind === 'OperationDefinition') {
      if (definition.operation === 'mutation' || definition.operation === 'subscription') {
        return false;
      }
    }
    
    // Don't batch real-time queries
    if (context?.realtime) {
      return false;
    }
    
    return true;
  },
  batchHttpLink,
  httpLink
);

// Optimized cache configuration
const cache = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        // Cache dashboard data for 5 minutes
        dashboardMetrics: {
          read(existing, { args, canRead }) {
            // Custom cache logic for dashboard metrics
            if (existing && canRead(existing)) {
              const cacheTime = existing.__timestamp;
              const now = Date.now();
              if (now - cacheTime < 5 * 60 * 1000) { // 5 minutes
                return existing;
              }
            }
            return undefined;
          },
        },
        // Implement cursor-based pagination caching
        userList: {
          keyArgs: ['filters'],
          merge(existing = [], incoming, { args }) {
            const { cursor } = args || {};
            if (cursor) {
              // Append for pagination
              return [...existing, ...incoming];
            }
            // Replace for new query
            return incoming;
          },
        },
      },
    },
    // Cache user data by ID
    User: {
      fields: {
        avatar: {
          read(url) {
            // Optimize image URLs for different screen sizes
            if (url && typeof url === 'string') {
              const dpr = window.devicePixelRatio || 1;
              const width = dpr > 1 ? 200 : 100;
              return `${url}?w=${width}&q=80`;
            }
            return url;
          },
        },
      },
    },
  },
  // Enable cache debugging in development
  ...(process.env.NODE_ENV === 'development' && {
    addTypename: true,
  }),
});

// Create Apollo Client with all optimizations
export const apolloClient = new ApolloClient({
  link: from([
    errorLink,
    authLink,
    persistedQueriesLink,
    splitLink,
  ]),
  cache,
  defaultOptions: {
    watchQuery: {
      // Optimize default query options
      errorPolicy: 'all',
      fetchPolicy: 'cache-first',
      nextFetchPolicy: 'cache-and-network',
    },
    query: {
      errorPolicy: 'all',
      fetchPolicy: 'cache-first',
    },
    mutate: {
      errorPolicy: 'all',
    },
  },
  // Enable query deduplication
  queryDeduplication: true,
  // Connect to DevTools in development
  connectToDevTools: process.env.NODE_ENV === 'development',
});

// Performance monitoring utilities
export const queryPerformanceMonitor = {
  startTiming: (operationName: string) => {
    const startTime = performance.now();
    return {
      end: () => {
        const duration = performance.now() - startTime;
        console.info(`GraphQL ${operationName}: ${duration.toFixed(2)}ms`);
        
        // Send metrics to monitoring service
        if (typeof window.gtag === 'function') {
          window.gtag('event', 'graphql_query', {
            event_category: 'performance',
            event_label: operationName,
            value: Math.round(duration),
          });
        }
        
        return duration;
      },
    };
  },
};

// Cache warming utility for critical queries
export const warmCache = async (queries: Array<{ query: any; variables?: any }>) => {
  const promises = queries.map(({ query, variables }) =>
    apolloClient.query({
      query,
      variables,
      fetchPolicy: 'cache-first',
    }).catch(error => {
      console.warn('Cache warming failed for query:', error);
    })
  );
  
  await Promise.allSettled(promises);
  console.info(`Cache warmed with ${queries.length} queries`);
};

// Extend the Window interface to include gtag
declare global {
  interface Window {
    gtag?: (event: string, action: string, params: Record<string, any>) => void;
  }
}