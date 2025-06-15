import { logger } from '../utils/logger';

export interface Activity {
  id: string;
  userId: string;
  type: 'LOGIN' | 'LOGOUT' | 'CREATE' | 'UPDATE' | 'DELETE' | 'VIEW' | 'EXPORT';
  message: string;
  action: string;
  resource: string;
  timestamp: Date;
  metadata?: Record<string, any>;
  ipAddress?: string;
  userAgent?: string;
}

export interface ActivitySummary {
  totalActivities: number;
  uniqueUsers: number;
  topActions: Array<{ action: string; count: number }>;
  activityByHour: Array<{ hour: number; count: number }>;
}

export class ActivityService {
  private activities: Activity[] = [];
  private readonly maxActivities = 10000; // Prevent memory issues

  constructor() {
    this.initializeMockActivities();
    // Clean up old activities every 10 minutes
    setInterval(() => this.cleanupOldActivities(), 10 * 60 * 1000);
  }

  private initializeMockActivities() {
    const actions = ['login', 'logout', 'view_dashboard', 'export_data', 'update_settings', 'create_report'];
    const resources = ['dashboard', 'user_profile', 'reports', 'settings', 'analytics'];
    const activityTypes: Array<'LOGIN' | 'LOGOUT' | 'CREATE' | 'UPDATE' | 'DELETE' | 'VIEW' | 'EXPORT'> = 
      ['LOGIN', 'LOGOUT', 'CREATE', 'UPDATE', 'DELETE', 'VIEW', 'EXPORT'];
    const userIds = ['1', '2', '3', '4', '5'];
    const now = new Date();

    // Generate mock activities for the last 24 hours
    for (let i = 0; i < 200; i++) {
      const timestamp = new Date(now.getTime() - (Math.random() * 24 * 60 * 60 * 1000));
      const action = actions[Math.floor(Math.random() * actions.length)];
      const resource = resources[Math.floor(Math.random() * resources.length)];
      const type = activityTypes[Math.floor(Math.random() * activityTypes.length)];
      
      const activity: Activity = {
        id: `activity_${i}`,
        userId: userIds[Math.floor(Math.random() * userIds.length)],
        type,
        message: `User performed ${action} on ${resource}`,
        action,
        resource,
        timestamp,
        metadata: {
          source: 'mock_data',
          sessionId: `session_${Math.floor(Math.random() * 50)}`
        },
        ipAddress: `192.168.1.${Math.floor(Math.random() * 255)}`,
        userAgent: 'Mozilla/5.0 (Mock Browser)'
      };

      this.activities.push(activity);
    }

    // Sort by timestamp (newest first)
    this.activities.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    
    logger.info(`ActivityService initialized with ${this.activities.length} mock activities`);
  }

  async recordActivity(activity: Omit<Activity, 'id' | 'timestamp'>): Promise<Activity> {
    const fullActivity: Activity = {
      ...activity,
      id: `activity_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      timestamp: new Date()
    };

    this.activities.unshift(fullActivity); // Add to beginning (newest first)

    // Keep only the most recent activities
    if (this.activities.length > this.maxActivities) {
      this.activities = this.activities.slice(0, this.maxActivities);
    }

    logger.debug(`Recorded activity: ${activity.action} on ${activity.resource} by user ${activity.userId}`);
    return fullActivity;
  }

  async getActivities(limit?: number, offset?: number): Promise<Activity[]> {
    const start = offset || 0;
    const end = limit ? start + limit : undefined;
    const result = this.activities.slice(start, end);
    
    logger.debug(`Retrieved ${result.length} activities (offset: ${start}, limit: ${limit})`);
    return result;
  }

  async getActivitiesByUser(userId: string, limit?: number): Promise<Activity[]> {
    const userActivities = this.activities.filter(activity => activity.userId === userId);
    const result = limit ? userActivities.slice(0, limit) : userActivities;
    
    logger.debug(`Retrieved ${result.length} activities for user: ${userId}`);
    return result;
  }

  async getActivitiesByAction(action: string, limit?: number): Promise<Activity[]> {
    const actionActivities = this.activities.filter(activity => activity.action === action);
    const result = limit ? actionActivities.slice(0, limit) : actionActivities;
    
    logger.debug(`Retrieved ${result.length} activities for action: ${action}`);
    return result;
  }

  async getRecentActivities(minutes: number = 60): Promise<Activity[]> {
    const cutoffTime = new Date(Date.now() - minutes * 60 * 1000);
    const recentActivities = this.activities.filter(
      activity => activity.timestamp >= cutoffTime
    );
    
    logger.debug(`Retrieved ${recentActivities.length} activities from last ${minutes} minutes`);
    return recentActivities;
  }

  async getActivitySummary(timeRange?: { start: Date; end: Date }): Promise<ActivitySummary> {
    let filteredActivities = this.activities;

    if (timeRange) {
      filteredActivities = this.activities.filter(
        activity => activity.timestamp >= timeRange.start && activity.timestamp <= timeRange.end
      );
    }

    // Calculate unique users
    const uniqueUsers = new Set(filteredActivities.map(a => a.userId)).size;

    // Calculate top actions
    const actionCounts = new Map<string, number>();
    filteredActivities.forEach(activity => {
      actionCounts.set(activity.action, (actionCounts.get(activity.action) || 0) + 1);
    });

    const topActions = Array.from(actionCounts.entries())
      .map(([action, count]) => ({ action, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    // Calculate activity by hour
    const hourCounts = new Array(24).fill(0);
    filteredActivities.forEach(activity => {
      const hour = activity.timestamp.getHours();
      hourCounts[hour]++;
    });

    const activityByHour = hourCounts.map((count, hour) => ({ hour, count }));

    const summary: ActivitySummary = {
      totalActivities: filteredActivities.length,
      uniqueUsers,
      topActions,
      activityByHour
    };

    logger.debug(`Generated activity summary: ${summary.totalActivities} activities, ${summary.uniqueUsers} unique users`);
    return summary;
  }

  private cleanupOldActivities() {
    const cutoffTime = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000); // 7 days ago
    const originalLength = this.activities.length;
    
    this.activities = this.activities.filter(activity => activity.timestamp > cutoffTime);
    
    const cleaned = originalLength - this.activities.length;
    if (cleaned > 0) {
      logger.info(`Cleaned up ${cleaned} old activities`);
    }
  }
}
