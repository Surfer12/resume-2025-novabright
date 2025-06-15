import { logger } from '../utils/logger';

export interface PerformanceMetric {
  id: string;
  operation: string;
  duration: number; // in milliseconds
  timestamp: Date;
  success: boolean;
  errorMessage?: string;
  metadata?: Record<string, any>;
}

export interface PerformanceStats {
  averageLatency: number;
  p95Latency: number;
  p99Latency: number;
  successRate: number;
  totalRequests: number;
  errorsCount: number;
}

export class PerformanceService {
  private metrics: PerformanceMetric[] = [];
  private readonly maxMetrics = 5000;

  constructor() {
    this.initializeMockMetrics();
    // Clean up old metrics every 15 minutes
    setInterval(() => this.cleanupOldMetrics(), 15 * 60 * 1000);
  }

  private initializeMockMetrics() {
    const operations = [
      'graphql_query',
      'database_read',
      'database_write',
      'cache_read',
      'cache_write',
      'api_call',
      'consciousness_calculation',
      'neural_network_render'
    ];

    const now = new Date();

    // Generate mock performance data for the last 2 hours
    for (let i = 0; i < 500; i++) {
      const timestamp = new Date(now.getTime() - (Math.random() * 2 * 60 * 60 * 1000));
      const operation = operations[Math.floor(Math.random() * operations.length)];
      
      // Simulate realistic latencies based on operation type
      let baseDuration = 50;
      if (operation.includes('database')) baseDuration = 150;
      if (operation.includes('consciousness')) baseDuration = 300;
      if (operation.includes('neural_network')) baseDuration = 500;
      
      const duration = baseDuration + (Math.random() * baseDuration * 2);
      const success = Math.random() > 0.05; // 95% success rate

      const metric: PerformanceMetric = {
        id: `perf_${i}`,
        operation,
        duration: Math.round(duration),
        timestamp,
        success,
        errorMessage: success ? undefined : 'Mock error for testing',
        metadata: {
          source: 'mock_data',
          version: '1.0',
          userId: Math.floor(Math.random() * 10).toString()
        }
      };

      this.metrics.push(metric);
    }

    // Sort by timestamp (newest first)
    this.metrics.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    
    logger.info(`PerformanceService initialized with ${this.metrics.length} mock metrics`);
  }

  async recordPerformance(metric: Omit<PerformanceMetric, 'id' | 'timestamp'>): Promise<PerformanceMetric> {
    const fullMetric: PerformanceMetric = {
      ...metric,
      id: `perf_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      timestamp: new Date()
    };

    this.metrics.unshift(fullMetric); // Add to beginning (newest first)

    // Keep only the most recent metrics
    if (this.metrics.length > this.maxMetrics) {
      this.metrics = this.metrics.slice(0, this.maxMetrics);
    }

    logger.debug(`Recorded performance: ${metric.operation} took ${metric.duration}ms (${metric.success ? 'success' : 'error'})`);
    return fullMetric;
  }

  async getPerformanceMetrics(operation?: string, limit?: number): Promise<PerformanceMetric[]> {
    let filteredMetrics = this.metrics;

    if (operation) {
      filteredMetrics = this.metrics.filter(metric => metric.operation === operation);
    }

    const result = limit ? filteredMetrics.slice(0, limit) : filteredMetrics;
    logger.debug(`Retrieved ${result.length} performance metrics${operation ? ` for operation: ${operation}` : ''}`);
    return result;
  }

  async getPerformanceStats(operation?: string, timeRange?: { start: Date; end: Date }): Promise<PerformanceStats> {
    let filteredMetrics = this.metrics;

    if (operation) {
      filteredMetrics = filteredMetrics.filter(metric => metric.operation === operation);
    }

    if (timeRange) {
      filteredMetrics = filteredMetrics.filter(
        metric => metric.timestamp >= timeRange.start && metric.timestamp <= timeRange.end
      );
    }

    if (filteredMetrics.length === 0) {
      return {
        averageLatency: 0,
        p95Latency: 0,
        p99Latency: 0,
        successRate: 0,
        totalRequests: 0,
        errorsCount: 0
      };
    }

    const durations = filteredMetrics.map(m => m.duration).sort((a, b) => a - b);
    const successfulRequests = filteredMetrics.filter(m => m.success);
    const errors = filteredMetrics.filter(m => !m.success);

    const stats: PerformanceStats = {
      averageLatency: Math.round(durations.reduce((sum, d) => sum + d, 0) / durations.length),
      p95Latency: Math.round(durations[Math.floor(durations.length * 0.95)] || 0),
      p99Latency: Math.round(durations[Math.floor(durations.length * 0.99)] || 0),
      successRate: Math.round((successfulRequests.length / filteredMetrics.length) * 100 * 100) / 100,
      totalRequests: filteredMetrics.length,
      errorsCount: errors.length
    };

    logger.debug(`Generated performance stats${operation ? ` for ${operation}` : ''}: avg=${stats.averageLatency}ms, p95=${stats.p95Latency}ms, success=${stats.successRate}%`);
    return stats;
  }

  async getRealtimePerformance(minutes: number = 5): Promise<PerformanceMetric[]> {
    const cutoffTime = new Date(Date.now() - minutes * 60 * 1000);
    const realtimeMetrics = this.metrics.filter(
      metric => metric.timestamp >= cutoffTime
    );
    
    logger.debug(`Retrieved ${realtimeMetrics.length} realtime performance metrics from last ${minutes} minutes`);
    return realtimeMetrics;
  }

  async getOperationTypes(): Promise<string[]> {
    const operations = Array.from(new Set(this.metrics.map(m => m.operation)));
    logger.debug(`Retrieved ${operations.length} operation types`);
    return operations;
  }

  // Helper method to measure and record performance automatically
  async measureOperation<T>(operation: string, fn: () => Promise<T>, metadata?: Record<string, any>): Promise<T> {
    const startTime = Date.now();
    let success = true;
    let errorMessage: string | undefined;
    let result: T;

    try {
      result = await fn();
    } catch (error) {
      success = false;
      errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw error;
    } finally {
      const duration = Date.now() - startTime;
      await this.recordPerformance({
        operation,
        duration,
        success,
        errorMessage,
        metadata
      });
    }

    return result;
  }

  private cleanupOldMetrics() {
    const cutoffTime = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24 hours ago
    const originalLength = this.metrics.length;
    
    this.metrics = this.metrics.filter(metric => metric.timestamp > cutoffTime);
    
    const cleaned = originalLength - this.metrics.length;
    if (cleaned > 0) {
      logger.info(`Cleaned up ${cleaned} old performance metrics`);
    }
  }
}
