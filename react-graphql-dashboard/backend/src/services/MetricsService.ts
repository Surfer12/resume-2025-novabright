import { logger } from '../utils/logger';

export interface Metric {
  id: string;
  name: string;
  value: number;
  unit: string;
  timestamp: Date;
  category: string;
  metadata?: Record<string, any>;
}

export interface MetricSummary {
  total: number;
  average: number;
  min: number;
  max: number;
  count: number;
}

export class MetricsService {
  private metrics: Map<string, Metric[]> = new Map();
  private readonly maxMetricsPerCategory = 1000; // Prevent memory issues

  constructor() {
    this.initializeMockMetrics();
    // Clean up old metrics every 5 minutes
    setInterval(() => this.cleanupOldMetrics(), 5 * 60 * 1000);
  }

  private initializeMockMetrics() {
    const categories = ['performance', 'consciousness', 'system', 'user_activity'];
    const now = new Date();

    categories.forEach(category => {
      const metrics: Metric[] = [];
      
      // Generate some historical data
      for (let i = 0; i < 50; i++) {
        const timestamp = new Date(now.getTime() - (i * 60 * 1000)); // Every minute
        metrics.push({
          id: `${category}_${i}`,
          name: `${category}_metric`,
          value: Math.random() * 100,
          unit: category === 'performance' ? 'ms' : 'score',
          timestamp,
          category,
          metadata: {
            source: 'mock_data',
            version: '1.0'
          }
        });
      }

      this.metrics.set(category, metrics);
    });

    logger.info(`MetricsService initialized with mock data for ${categories.length} categories`);
  }

  async recordMetric(metric: Omit<Metric, 'id' | 'timestamp'>): Promise<Metric> {
    const fullMetric: Metric = {
      ...metric,
      id: `${metric.category}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      timestamp: new Date()
    };

    const categoryMetrics = this.metrics.get(metric.category) || [];
    categoryMetrics.push(fullMetric);

    // Keep only the most recent metrics to prevent memory issues
    if (categoryMetrics.length > this.maxMetricsPerCategory) {
      categoryMetrics.splice(0, categoryMetrics.length - this.maxMetricsPerCategory);
    }

    this.metrics.set(metric.category, categoryMetrics);
    logger.debug(`Recorded metric: ${metric.name} = ${metric.value} ${metric.unit}`);
    
    return fullMetric;
  }

  async getMetrics(category: string, limit?: number): Promise<Metric[]> {
    const categoryMetrics = this.metrics.get(category) || [];
    const result = limit ? categoryMetrics.slice(-limit) : categoryMetrics;
    
    logger.debug(`Retrieved ${result.length} metrics for category: ${category}`);
    return result;
  }

  async getMetricsSummary(category: string, timeRange?: { start: Date; end: Date }): Promise<MetricSummary> {
    let categoryMetrics = this.metrics.get(category) || [];

    // Filter by time range if provided
    if (timeRange) {
      categoryMetrics = categoryMetrics.filter(
        metric => metric.timestamp >= timeRange.start && metric.timestamp <= timeRange.end
      );
    }

    if (categoryMetrics.length === 0) {
      return { total: 0, average: 0, min: 0, max: 0, count: 0 };
    }

    const values = categoryMetrics.map(m => m.value);
    const summary: MetricSummary = {
      total: values.reduce((sum, val) => sum + val, 0),
      average: values.reduce((sum, val) => sum + val, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length
    };

    logger.debug(`Generated summary for ${category}: avg=${summary.average.toFixed(2)}, count=${summary.count}`);
    return summary;
  }

  async getRealtimeMetrics(category: string): Promise<Metric[]> {
    const now = new Date();
    const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);
    
    const categoryMetrics = this.metrics.get(category) || [];
    const realtimeMetrics = categoryMetrics.filter(
      metric => metric.timestamp >= fiveMinutesAgo
    );

    logger.debug(`Retrieved ${realtimeMetrics.length} realtime metrics for category: ${category}`);
    return realtimeMetrics;
  }

  async getAllCategories(): Promise<string[]> {
    const categories = Array.from(this.metrics.keys());
    logger.debug(`Retrieved ${categories.length} metric categories`);
    return categories;
  }

  private cleanupOldMetrics() {
    const cutoffTime = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24 hours ago
    let totalCleaned = 0;

    this.metrics.forEach((metrics, category) => {
      const originalLength = metrics.length;
      const filteredMetrics = metrics.filter(metric => metric.timestamp > cutoffTime);
      
      if (filteredMetrics.length !== originalLength) {
        this.metrics.set(category, filteredMetrics);
        totalCleaned += originalLength - filteredMetrics.length;
      }
    });

    if (totalCleaned > 0) {
      logger.info(`Cleaned up ${totalCleaned} old metrics`);
    }
  }
}
