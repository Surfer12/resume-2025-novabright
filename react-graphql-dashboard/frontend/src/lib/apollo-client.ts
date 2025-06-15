import {
  ApolloClient,
  InMemoryCache,
  createHttpLink,
  from,
} from '@apollo/client';
import { setContext } from '@apollo/client/link/context';
// import { createPersistedQueryLink } from '@apollo/client/link/persisted-queries'; // Temporarily disabled
// import { sha256 } from 'crypto-hash'; // Temporarily disabled
import { onError } from '@apollo/client/link/error';

// Environment configuration
const API_ENDPOINT = process.env.VITE_GRAPHQL_ENDPOINT || 'http://localhost:4000/graphql';

// Performance optimization: Persisted queries for reduced bandwidth - TEMPORARILY DISABLED
// const persistedQueriesLink = createPersistedQueryLink({
//   sha256,
//   useGETForHashedQueries: false, // Use POST for persisted queries
// });

// Regular HTTP link for all operations (no batching, no persisted queries)
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

// Create Apollo Client with simplified configuration
export const apolloClient = new ApolloClient({
  link: from([
    errorLink,
    authLink,
    httpLink, // Simple HTTP link only
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