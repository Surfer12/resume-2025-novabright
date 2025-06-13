#!/usr/bin/env node

const https = require('https');
const http = require('http');
const { performance } = require('perf_hooks');

// Configuration
const config = {
  // Test endpoints
  optimizedEndpoint: process.env.OPTIMIZED_ENDPOINT || 'https://api.dashboard.com/graphql',
  baselineEndpoint: process.env.BASELINE_ENDPOINT || 'https://baseline-api.com/graphql',
  
  // Test parameters
  concurrentUsers: parseInt(process.env.CONCURRENT_USERS) || 50,
  requestsPerUser: parseInt(process.env.REQUESTS_PER_USER) || 20,
  warmupRequests: parseInt(process.env.WARMUP_REQUESTS) || 10,
  testDuration: parseInt(process.env.TEST_DURATION) || 300000, // 5 minutes
  
  // Request configuration
  timeout: 30000,
  retries: 3,
};

// Test queries for different scenarios
const testQueries = {
  // Simple dashboard query
  dashboardBasic: {
    query: `
      query GetDashboardData($timeRange: TimeRange!) {
        dashboardMetrics(timeRange: $timeRange, metrics: [CPU, MEMORY, NETWORK]) {
          id
          type
          value
          change
          changePercent
          timestamp
        }
        systemStatus {
          cpu
          memory
          storage
          network
          healthy
        }
      }
    `,
    variables: { timeRange: 'DAY' },
    weight: 0.4, // 40% of requests
  },
  
  // Complex query with relationships
  dashboardComplex: {
    query: `
      query GetComplexDashboard($timeRange: TimeRange!) {
        dashboardMetrics(timeRange: $timeRange, metrics: [CPU, MEMORY, NETWORK, STORAGE, REQUESTS]) {
          id
          type
          value
          change
          changePercent
          timestamp
          trend {
            timestamp
            value
          }
        }
        recentActivity(limit: 20) {
          id
          type
          message
          timestamp
          user {
            id
            name
            email
            role
          }
        }
        users(first: 10) {
          edges {
            node {
              id
              name
              email
              role
              lastActive
              statistics {
                loginCount
                activeProjects
              }
            }
          }
        }
      }
    `,
    variables: { timeRange: 'DAY' },
    weight: 0.3, // 30% of requests
  },
  
  // Search query
  search: {
    query: `
      query Search($query: String!, $limit: Int!) {
        search(query: $query, limit: $limit) {
          ... on User {
            id
            name
            email
            role
          }
          ... on Project {
            id
            name
            description
            status
          }
        }
      }
    `,
    variables: { query: 'test', limit: 10 },
    weight: 0.2, // 20% of requests
  },
  
  // Performance metrics query
  performanceMetrics: {
    query: `
      query GetPerformanceMetrics($timeRange: TimeRange!) {
        performanceMetrics(timeRange: $timeRange) {
          queryLatency {
            average
            p95
            p99
          }
          cacheHitRate {
            overall
          }
          errorRate {
            rate
          }
          throughput {
            requestsPerSecond
          }
        }
      }
    `,
    variables: { timeRange: 'HOUR' },
    weight: 0.1, // 10% of requests
  },
};

// Performance metrics collector
class PerformanceCollector {
  constructor() {
    this.reset();
  }

  reset() {
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      totalLatency: 0,
      latencies: [],
      errors: [],
      cacheHits: 0,
      cacheMisses: 0,
      startTime: null,
      endTime: null,
    };
  }

  recordRequest(latency, success, cacheHit = false, error = null) {
    this.metrics.totalRequests++;
    this.metrics.totalLatency += latency;
    this.metrics.latencies.push(latency);
    
    if (success) {
      this.metrics.successfulRequests++;
    } else {
      this.metrics.failedRequests++;
      if (error) this.metrics.errors.push(error);
    }
    
    if (cacheHit) {
      this.metrics.cacheHits++;
    } else {
      this.metrics.cacheMisses++;
    }
  }

  getStats() {
    const { latencies } = this.metrics;
    latencies.sort((a, b) => a - b);
    
    const duration = this.metrics.endTime - this.metrics.startTime;
    
    return {
      totalRequests: this.metrics.totalRequests,
      successfulRequests: this.metrics.successfulRequests,
      failedRequests: this.metrics.failedRequests,
      successRate: (this.metrics.successfulRequests / this.metrics.totalRequests * 100).toFixed(2),
      
      // Latency statistics
      averageLatency: this.metrics.totalLatency / this.metrics.totalRequests,
      medianLatency: latencies[Math.floor(latencies.length / 2)],
      p95Latency: latencies[Math.floor(latencies.length * 0.95)],
      p99Latency: latencies[Math.floor(latencies.length * 0.99)],
      minLatency: Math.min(...latencies),
      maxLatency: Math.max(...latencies),
      
      // Throughput
      requestsPerSecond: (this.metrics.totalRequests / (duration / 1000)).toFixed(2),
      
      // Cache performance
      cacheHitRate: this.metrics.cacheHits / (this.metrics.cacheHits + this.metrics.cacheMisses) * 100,
      
      // Duration
      testDuration: duration,
      
      // Errors
      errorRate: (this.metrics.failedRequests / this.metrics.totalRequests * 100).toFixed(2),
      errors: this.metrics.errors,
    };
  }
}

// HTTP request function with performance monitoring
function makeRequest(endpoint, query, variables = {}) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({ query, variables });
    const url = new URL(endpoint);
    
    const options = {
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data),
        'User-Agent': 'GraphQL-Performance-Benchmark/1.0',
        'X-Request-ID': `benchmark-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      },
      timeout: config.timeout,
    };

    const startTime = performance.now();
    const client = url.protocol === 'https:' ? https : http;
    
    const req = client.request(options, (res) => {
      let responseData = '';
      
      res.on('data', (chunk) => {
        responseData += chunk;
      });
      
      res.on('end', () => {
        const endTime = performance.now();
        const latency = endTime - startTime;
        
        try {
          const result = JSON.parse(responseData);
          const cacheHit = res.headers['x-cache-status'] === 'HIT';
          
          resolve({
            latency,
            success: res.statusCode === 200 && !result.errors,
            cacheHit,
            data: result,
            statusCode: res.statusCode,
            headers: res.headers,
          });
        } catch (error) {
          resolve({
            latency,
            success: false,
            cacheHit: false,
            error: `Parse error: ${error.message}`,
            statusCode: res.statusCode,
          });
        }
      });
    });

    req.on('error', (error) => {
      const endTime = performance.now();
      const latency = endTime - startTime;
      
      resolve({
        latency,
        success: false,
        cacheHit: false,
        error: `Request error: ${error.message}`,
      });
    });

    req.on('timeout', () => {
      req.destroy();
      const endTime = performance.now();
      const latency = endTime - startTime;
      
      resolve({
        latency,
        success: false,
        cacheHit: false,
        error: 'Request timeout',
      });
    });

    req.write(data);
    req.end();
  });
}

// Generate weighted random query
function getRandomQuery() {
  const rand = Math.random();
  let cumulativeWeight = 0;
  
  for (const [name, queryConfig] of Object.entries(testQueries)) {
    cumulativeWeight += queryConfig.weight;
    if (rand <= cumulativeWeight) {
      return { name, ...queryConfig };
    }
  }
  
  // Fallback to first query
  const fallback = Object.entries(testQueries)[0];
  return { name: fallback[0], ...fallback[1] };
}

// Run single user simulation
async function runUserSimulation(endpoint, collector, userId) {
  console.log(`Starting user ${userId} simulation against ${endpoint}`);
  
  for (let i = 0; i < config.requestsPerUser; i++) {
    const { name, query, variables } = getRandomQuery();
    
    try {
      const result = await makeRequest(endpoint, query, variables);
      collector.recordRequest(
        result.latency,
        result.success,
        result.cacheHit,
        result.error
      );
      
      // Add small random delay between requests (realistic user behavior)
      const delay = Math.random() * 1000 + 500; // 500-1500ms
      await new Promise(resolve => setTimeout(resolve, delay));
      
    } catch (error) {
      collector.recordRequest(0, false, false, error.message);
    }
  }
  
  console.log(`User ${userId} completed ${config.requestsPerUser} requests`);
}

// Warmup function
async function warmupEndpoint(endpoint) {
  console.log(`Warming up endpoint: ${endpoint}`);
  
  const promises = [];
  for (let i = 0; i < config.warmupRequests; i++) {
    const { query, variables } = getRandomQuery();
    promises.push(makeRequest(endpoint, query, variables));
  }
  
  await Promise.allSettled(promises);
  console.log(`Warmup completed with ${config.warmupRequests} requests`);
}

// Run benchmark against a single endpoint
async function runBenchmark(endpoint, label) {
  console.log(`\n🚀 Starting benchmark for ${label}: ${endpoint}`);
  
  // Warmup
  await warmupEndpoint(endpoint);
  
  // Initialize collector
  const collector = new PerformanceCollector();
  collector.metrics.startTime = performance.now();
  
  // Create user simulations
  const userPromises = [];
  for (let i = 0; i < config.concurrentUsers; i++) {
    userPromises.push(runUserSimulation(endpoint, collector, i + 1));
  }
  
  // Wait for all users to complete
  console.log(`Running ${config.concurrentUsers} concurrent users...`);
  await Promise.allSettled(userPromises);
  
  collector.metrics.endTime = performance.now();
  
  const stats = collector.getStats();
  
  console.log(`\n📊 Results for ${label}:`);
  console.log(`Total Requests: ${stats.totalRequests}`);
  console.log(`Success Rate: ${stats.successRate}%`);
  console.log(`Average Latency: ${stats.averageLatency.toFixed(2)}ms`);
  console.log(`Median Latency: ${stats.medianLatency.toFixed(2)}ms`);
  console.log(`95th Percentile: ${stats.p95Latency.toFixed(2)}ms`);
  console.log(`99th Percentile: ${stats.p99Latency.toFixed(2)}ms`);
  console.log(`Requests/Second: ${stats.requestsPerSecond}`);
  console.log(`Cache Hit Rate: ${stats.cacheHitRate.toFixed(2)}%`);
  console.log(`Error Rate: ${stats.errorRate}%`);
  
  if (stats.errors.length > 0) {
    console.log(`Errors: ${stats.errors.slice(0, 5).join(', ')}${stats.errors.length > 5 ? '...' : ''}`);
  }
  
  return stats;
}

// Compare results and calculate improvement
function compareResults(optimizedStats, baselineStats) {
  console.log('\n📈 PERFORMANCE COMPARISON');
  console.log('=' .repeat(50));
  
  const latencyImprovement = ((baselineStats.averageLatency - optimizedStats.averageLatency) / baselineStats.averageLatency * 100);
  const throughputImprovement = ((optimizedStats.requestsPerSecond - baselineStats.requestsPerSecond) / baselineStats.requestsPerSecond * 100);
  const p95Improvement = ((baselineStats.p95Latency - optimizedStats.p95Latency) / baselineStats.p95Latency * 100);
  
  console.log(`Average Latency Improvement: ${latencyImprovement.toFixed(1)}%`);
  console.log(`  Baseline: ${baselineStats.averageLatency.toFixed(2)}ms`);
  console.log(`  Optimized: ${optimizedStats.averageLatency.toFixed(2)}ms`);
  
  console.log(`\n95th Percentile Improvement: ${p95Improvement.toFixed(1)}%`);
  console.log(`  Baseline: ${baselineStats.p95Latency.toFixed(2)}ms`);
  console.log(`  Optimized: ${optimizedStats.p95Latency.toFixed(2)}ms`);
  
  console.log(`\nThroughput Improvement: ${throughputImprovement.toFixed(1)}%`);
  console.log(`  Baseline: ${baselineStats.requestsPerSecond} req/s`);
  console.log(`  Optimized: ${optimizedStats.requestsPerSecond} req/s`);
  
  console.log(`\nCache Performance:`);
  console.log(`  Optimized Cache Hit Rate: ${optimizedStats.cacheHitRate.toFixed(1)}%`);
  console.log(`  Baseline Cache Hit Rate: ${baselineStats.cacheHitRate.toFixed(1)}%`);
  
  // Verify 30% improvement target
  const targetImprovement = 30;
  if (latencyImprovement >= targetImprovement) {
    console.log(`\n✅ SUCCESS: Achieved ${latencyImprovement.toFixed(1)}% latency improvement (target: ${targetImprovement}%)`);
  } else {
    console.log(`\n❌ TARGET MISSED: Only achieved ${latencyImprovement.toFixed(1)}% improvement (target: ${targetImprovement}%)`);
  }
  
  return {
    latencyImprovement,
    throughputImprovement,
    p95Improvement,
    targetMet: latencyImprovement >= targetImprovement,
  };
}

// Generate detailed report
function generateReport(optimizedStats, baselineStats, comparison) {
  const report = {
    timestamp: new Date().toISOString(),
    testConfiguration: config,
    results: {
      optimized: optimizedStats,
      baseline: baselineStats,
      comparison,
    },
    summary: {
      latencyImprovementPercent: comparison.latencyImprovement,
      throughputImprovementPercent: comparison.throughputImprovement,
      targetAchieved: comparison.targetMet,
      optimizedAverageLatency: optimizedStats.averageLatency,
      baselineAverageLatency: baselineStats.averageLatency,
    },
  };
  
  return report;
}

// Main benchmark function
async function main() {
  try {
    console.log('🎯 React-GraphQL Dashboard Performance Benchmark');
    console.log('=' .repeat(60));
    console.log(`Configuration:`);
    console.log(`  Concurrent Users: ${config.concurrentUsers}`);
    console.log(`  Requests per User: ${config.requestsPerUser}`);
    console.log(`  Total Requests: ${config.concurrentUsers * config.requestsPerUser}`);
    console.log(`  Warmup Requests: ${config.warmupRequests}`);
    
    // Run baseline benchmark
    const baselineStats = await runBenchmark(config.baselineEndpoint, 'BASELINE');
    
    // Small delay between tests
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Run optimized benchmark
    const optimizedStats = await runBenchmark(config.optimizedEndpoint, 'OPTIMIZED');
    
    // Compare results
    const comparison = compareResults(optimizedStats, baselineStats);
    
    // Generate and save report
    const report = generateReport(optimizedStats, baselineStats, comparison);
    
    const reportPath = `benchmark-report-${Date.now()}.json`;
    require('fs').writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 Detailed report saved to: ${reportPath}`);
    
    // Exit with appropriate code
    process.exit(comparison.targetMet ? 0 : 1);
    
  } catch (error) {
    console.error('❌ Benchmark failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = {
  runBenchmark,
  PerformanceCollector,
  makeRequest,
  compareResults,
};