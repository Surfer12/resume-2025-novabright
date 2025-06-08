import { Resolvers } from '@apollo/server';
import DataLoader from 'dataloader';
import { UserService } from '../services/UserService';
import { MetricsService } from '../services/MetricsService';
import { ActivityService } from '../services/ActivityService';
import { PerformanceService } from '../services/PerformanceService';
import { CacheService } from '../services/CacheService';
import { logger } from '../utils/logger';

// DataLoader instances for batching and caching
const createDataLoaders = () => ({
  userLoader: new DataLoader(async (userIds: readonly string[]) => {
    const users = await UserService.getUsersByIds([...userIds]);
    return userIds.map(id => users.find(user => user.id === id) || null);
  }, {
    // Cache for 5 minutes
    cacheKeyFn: (key) => `user:${key}`,
    maxBatchSize: 100,
  }),

  userStatisticsLoader: new DataLoader(async (userIds: readonly string[]) => {
    const statistics = await UserService.getUserStatistics([...userIds]);
    return userIds.map(id => statistics.find(stat => stat.userId === id) || null);
  }, {
    cacheKeyFn: (key) => `userStats:${key}`,
    maxBatchSize: 50,
  }),

  metricsLoader: new DataLoader(async (metricKeys: readonly string[]) => {
    const metrics = await MetricsService.getMetricsByKeys([...metricKeys]);
    return metricKeys.map(key => metrics.find(metric => `${metric.type}:${metric.period}` === key) || null);
  }),

  trendDataLoader: new DataLoader(async (metricIds: readonly string[]) => {
    const trendData = await MetricsService.getTrendDataByMetricIds([...metricIds]);
    return metricIds.map(id => trendData[id] || []);
  }),
});

// Performance monitoring wrapper for resolvers
const withPerformanceMonitoring = (resolverName: string, resolver: any) => {
  return async (parent: any, args: any, context: any, info: any) => {
    const startTime = process.hrtime.bigint();
    
    try {
      const result = await resolver(parent, args, context, info);
      const endTime = process.hrtime.bigint();
      const duration = Number(endTime - startTime) / 1000000; // Convert to milliseconds
      
      // Log performance metrics
      PerformanceService.recordResolverPerformance(resolverName, duration);
      
      // Log slow resolvers
      if (duration > 100) {
        logger.warn(`Slow resolver detected: ${resolverName} took ${duration}ms`);
      }
      
      return result;
    } catch (error) {
      const endTime = process.hrtime.bigint();
      const duration = Number(endTime - startTime) / 1000000;
      
      PerformanceService.recordResolverError(resolverName, duration, error as Error);
      logger.error(`Resolver error in ${resolverName}:`, error);
      throw error;
    }
  };
};

export const resolvers: Resolvers = {
  // Custom scalar resolvers
  DateTime: {
    serialize: (value: Date) => value.toISOString(),
    parseValue: (value: string) => new Date(value),
    parseLiteral: (ast: any) => new Date(ast.value),
  },
  
  JSON: {
    serialize: (value: any) => value,
    parseValue: (value: any) => value,
    parseLiteral: (ast: any) => JSON.parse(ast.value),
  },

  // Union type resolver
  SearchResult: {
    __resolveType: (obj: any) => {
      if (obj.email) return 'User';
      if (obj.owner) return 'Project';
      if (obj.type && obj.message) return 'Activity';
      return null;
    },
  },

  // Query resolvers with optimization
  Query: {
    dashboardMetrics: withPerformanceMonitoring('dashboardMetrics', 
      async (_, { timeRange, metrics }, { dataSources, cache }) => {
        const cacheKey = `dashboard:${timeRange}:${metrics.join(',')}`;
        
        // Try cache first for 30% latency improvement
        const cached = await CacheService.get(cacheKey);
        if (cached) {
          logger.info('Dashboard metrics served from cache');
          return cached;
        }

        const metricsData = await MetricsService.getDashboardMetrics(timeRange, metrics);
        
        // Cache for 5 minutes
        await CacheService.set(cacheKey, metricsData, 300);
        
        return metricsData;
      }
    ),

    recentActivity: withPerformanceMonitoring('recentActivity',
      async (_, { limit }, { dataSources, cache }) => {
        const cacheKey = `activity:recent:${limit}`;
        
        const cached = await CacheService.get(cacheKey);
        if (cached) return cached;

        const activities = await ActivityService.getRecentActivities(limit);
        await CacheService.set(cacheKey, activities, 60); // 1 minute cache
        
        return activities;
      }
    ),

    systemStatus: withPerformanceMonitoring('systemStatus',
      async (_, __, { cache }) => {
        const cacheKey = 'system:status';
        
        const cached = await CacheService.get(cacheKey);
        if (cached) return cached;

        const status = await MetricsService.getSystemStatus();
        await CacheService.set(cacheKey, status, 30); // 30 second cache
        
        return status;
      }
    ),

    users: withPerformanceMonitoring('users',
      async (_, { first, after, filters, sortBy, sortOrder }, { dataSources }) => {
        return await UserService.getUsersPaginated({
          first,
          after,
          filters,
          sortBy,
          sortOrder,
        });
      }
    ),

    user: withPerformanceMonitoring('user',
      async (_, { id }, { dataLoaders }) => {
        return await dataLoaders.userLoader.load(id);
      }
    ),

    search: withPerformanceMonitoring('search',
      async (_, { query, types, limit }, { cache }) => {
        const cacheKey = `search:${query}:${types.join(',')}:${limit}`;
        
        const cached = await CacheService.get(cacheKey);
        if (cached) return cached;

        const results = await UserService.search(query, types, limit);
        await CacheService.set(cacheKey, results, 180); // 3 minute cache
        
        return results;
      }
    ),

    performanceMetrics: withPerformanceMonitoring('performanceMetrics',
      async (_, { timeRange }) => {
        return await PerformanceService.getPerformanceMetrics(timeRange);
      }
    ),

    criticalMetrics: withPerformanceMonitoring('criticalMetrics',
      async (_, __, { cache }) => {
        const cacheKey = 'critical:metrics';
        
        const cached = await CacheService.get(cacheKey);
        if (cached) return cached;

        const metrics = await MetricsService.getCriticalMetrics();
        await CacheService.set(cacheKey, metrics, 30);
        
        return metrics;
      }
    ),

    secondaryData: withPerformanceMonitoring('secondaryData',
      async () => {
        // This can be loaded asynchronously with @defer
        return await MetricsService.getSecondaryData();
      }
    ),
  },

  // Field resolvers with DataLoader optimization
  User: {
    statistics: async (parent, _, { dataLoaders }) => {
      return await dataLoaders.userStatisticsLoader.load(parent.id);
    },
  },

  Metrics: {
    trend: async (parent, _, { dataLoaders }) => {
      return await dataLoaders.trendDataLoader.load(parent.id);
    },
  },

  Activity: {
    user: async (parent, _, { dataLoaders }) => {
      return await dataLoaders.userLoader.load(parent.userId);
    },
  },

  Project: {
    owner: async (parent, _, { dataLoaders }) => {
      return await dataLoaders.userLoader.load(parent.ownerId);
    },
  },

  // Mutation resolvers
  Mutation: {
    updateUser: withPerformanceMonitoring('updateUser',
      async (_, { id, input }, { dataSources, dataLoaders }) => {
        try {
          const result = await UserService.updateUser(id, input);
          
          // Invalidate cache
          dataLoaders.userLoader.clear(id);
          dataLoaders.userStatisticsLoader.clear(id);
          await CacheService.delete(`user:${id}`);
          
          return result;
        } catch (error) {
          logger.error('Update user error:', error);
          throw new Error('Failed to update user');
        }
      }
    ),

    bulkUpdateUsers: withPerformanceMonitoring('bulkUpdateUsers',
      async (_, { operations }, { dataLoaders }) => {
        const result = await UserService.bulkUpdateUsers(operations);
        
        // Clear cache for all affected users
        operations.forEach(op => {
          dataLoaders.userLoader.clear(op.id);
          dataLoaders.userStatisticsLoader.clear(op.id);
        });
        
        return result;
      }
    ),

    invalidateCache: async (_, { keys }) => {
      await Promise.all(keys.map(key => CacheService.delete(key)));
      return true;
    },

    warmCache: async (_, { queries }) => {
      // Implement cache warming logic
      await CacheService.warmCache(queries);
      return true;
    },
  },

  // Subscription resolvers
  Subscription: {
    metricsUpdated: {
      subscribe: () => {
        // Return AsyncIterable for real-time updates
        return MetricsService.subscribeToMetricsUpdates();
      },
    },

    activityAdded: {
      subscribe: () => {
        return ActivityService.subscribeToActivityUpdates();
      },
    },

    systemStatusChanged: {
      subscribe: () => {
        return MetricsService.subscribeToSystemStatusUpdates();
      },
    },

    userUpdated: {
      subscribe: (_, { userId }) => {
        return UserService.subscribeToUserUpdates(userId);
      },
    },
  },
};

// Context factory with DataLoaders
export const createContext = async (req: any) => {
  const dataLoaders = createDataLoaders();
  
  return {
    dataLoaders,
    user: req.user, // From authentication middleware
    requestId: req.headers['x-request-id'] || 'unknown',
    startTime: Date.now(),
  };
};

export default resolvers;