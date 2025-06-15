import { PubSub } from 'graphql-subscriptions';
import { logger } from '../utils/logger';

// Create a new PubSub instance with proper typing
const pubsub = new PubSub();

// Constants for subscription events
export const EVENTS = {
  METRICS_UPDATED: 'METRICS_UPDATED',
  ACTIVITY_ADDED: 'ACTIVITY_ADDED',
  SYSTEM_STATUS_CHANGED: 'SYSTEM_STATUS_CHANGED',
  USER_UPDATED: 'USER_UPDATED',
};

// Helper function to publish metrics updates
export const publishMetricsUpdate = (metrics: any) => {
  pubsub.publish(EVENTS.METRICS_UPDATED, { metricsUpdated: metrics });
};

// Helper function to publish activity updates
export const publishActivityUpdate = (activity: any) => {
  pubsub.publish(EVENTS.ACTIVITY_ADDED, { activityAdded: activity });
};

// Helper function to publish system status updates
export const publishSystemStatusUpdate = (status: any) => {
  pubsub.publish(EVENTS.SYSTEM_STATUS_CHANGED, { systemStatusChanged: status });
};

// Helper function to publish user updates
export const publishUserUpdate = (userId: string, user: any) => {
  pubsub.publish(`${EVENTS.USER_UPDATED}.${userId}`, { userUpdated: user });
};

// Subscription resolvers with proper typing
export const subscriptionResolvers = {
  Subscription: {
    metricsUpdated: {
      subscribe: () => pubsub.asyncIterator([EVENTS.METRICS_UPDATED]),
    },
    
    activityAdded: {
      subscribe: () => pubsub.asyncIterator([EVENTS.ACTIVITY_ADDED]),
    },
    
    systemStatusChanged: {
      subscribe: () => pubsub.asyncIterator([EVENTS.SYSTEM_STATUS_CHANGED]),
    },
    
    userUpdated: {
      subscribe: (_: any, { userId }: { userId: string }) => {
        logger.info(`User ${userId} subscribed to updates`);
        return pubsub.asyncIterator([`${EVENTS.USER_UPDATED}.${userId}`]);
      },
    },
  },
};

// Simulated data generation for subscriptions (for development)
export const startSubscriptionSimulation = (metricsService: any, activityService: any) => {
  // Simulate metrics updates every 5 seconds
  setInterval(() => {
    metricsService.getMetrics('performance', 1).then((metrics: any[]) => {
      if (metrics && metrics.length > 0) {
        publishMetricsUpdate(metrics[0]);
      }
    }).catch((error: any) => {
      logger.error('Error simulating metrics update', error);
    });
  }, 5000);

  // Simulate activity updates every 8 seconds
  setInterval(() => {
    activityService.getRecentActivities(1).then((activities: any[]) => {
      if (activities && activities.length > 0) {
        publishActivityUpdate(activities[0]);
      }
    }).catch((error: any) => {
      logger.error('Error simulating activity update', error);
    });
  }, 8000);

  // Simulate system status updates every 10 seconds
  setInterval(() => {
    const systemStatus = {
      cpu: Math.random() * 100,
      memory: Math.random() * 100,
      storage: Math.random() * 100,
      network: Math.random() * 100,
      healthy: true,
      lastUpdated: new Date().toISOString(),
    };
    
    publishSystemStatusUpdate(systemStatus);
  }, 10000);

  logger.info('Subscription simulation started');
};

export default subscriptionResolvers; 