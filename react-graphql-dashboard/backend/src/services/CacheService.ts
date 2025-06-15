import { logger } from '../utils/logger';

export interface CacheEntry<T = any> {
  key: string;
  value: T;
  timestamp: Date;
  ttl: number; // Time to live in milliseconds
  hits: number;
}

export interface CacheStats {
  totalKeys: number;
  totalHits: number;
  totalMisses: number;
  hitRate: number;
  memoryUsage: number; // Approximate memory usage in bytes
}

export class CacheService {
  private cache: Map<string, CacheEntry> = new Map();
  private stats = {
    hits: 0,
    misses: 0
  };

  constructor() {
    // Clean up expired entries every 5 minutes
    setInterval(() => this.cleanupExpiredEntries(), 5 * 60 * 1000);
    logger.info('CacheService initialized');
  }

  async set<T>(key: string, value: T, ttlSeconds: number = 300): Promise<void> {
    const entry: CacheEntry<T> = {
      key,
      value,
      timestamp: new Date(),
      ttl: ttlSeconds * 1000, // Convert to milliseconds
      hits: 0
    };

    this.cache.set(key, entry);
    logger.debug(`Cache SET: ${key} (TTL: ${ttlSeconds}s)`);
  }

  async get<T>(key: string): Promise<T | null> {
    const entry = this.cache.get(key);

    if (!entry) {
      this.stats.misses++;
      logger.debug(`Cache MISS: ${key}`);
      return null;
    }

    // Check if entry has expired
    const now = Date.now();
    const entryAge = now - entry.timestamp.getTime();

    if (entryAge > entry.ttl) {
      this.cache.delete(key);
      this.stats.misses++;
      logger.debug(`Cache EXPIRED: ${key} (age: ${entryAge}ms, ttl: ${entry.ttl}ms)`);
      return null;
    }

    // Update hit count and stats
    entry.hits++;
    this.stats.hits++;
    logger.debug(`Cache HIT: ${key} (hits: ${entry.hits})`);
    
    return entry.value as T;
  }

  async has(key: string): Promise<boolean> {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return false;
    }

    // Check if entry has expired
    const now = Date.now();
    const entryAge = now - entry.timestamp.getTime();

    if (entryAge > entry.ttl) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  async delete(key: string): Promise<boolean> {
    const deleted = this.cache.delete(key);
    if (deleted) {
      logger.debug(`Cache DELETE: ${key}`);
    }
    return deleted;
  }

  async clear(): Promise<void> {
    const size = this.cache.size;
    this.cache.clear();
    this.stats.hits = 0;
    this.stats.misses = 0;
    logger.info(`Cache CLEAR: removed ${size} entries`);
  }

  async getStats(): Promise<CacheStats> {
    const totalRequests = this.stats.hits + this.stats.misses;
    const hitRate = totalRequests > 0 ? (this.stats.hits / totalRequests) * 100 : 0;

    // Approximate memory usage calculation
    let memoryUsage = 0;
    this.cache.forEach(entry => {
      // Rough estimation: key size + JSON string size of value + metadata overhead
      const keySize = entry.key.length * 2; // UTF-16 characters
      const valueSize = JSON.stringify(entry.value).length * 2;
      const metadataSize = 100; // Approximate overhead for timestamp, ttl, hits
      memoryUsage += keySize + valueSize + metadataSize;
    });

    const stats: CacheStats = {
      totalKeys: this.cache.size,
      totalHits: this.stats.hits,
      totalMisses: this.stats.misses,
      hitRate: Math.round(hitRate * 100) / 100,
      memoryUsage
    };

    logger.debug(`Cache stats: ${stats.totalKeys} keys, ${stats.hitRate}% hit rate, ${Math.round(stats.memoryUsage / 1024)}KB memory`);
    return stats;
  }

  async getKeys(): Promise<string[]> {
    return Array.from(this.cache.keys());
  }

  async getEntries(): Promise<CacheEntry[]> {
    return Array.from(this.cache.values());
  }

  // Helper method for caching function results
  async memoize<T>(
    key: string,
    fn: () => Promise<T>,
    ttlSeconds: number = 300
  ): Promise<T> {
    // Try to get from cache first
    const cached = await this.get<T>(key);
    if (cached !== null) {
      return cached;
    }

    // Execute function and cache result
    const result = await fn();
    await this.set(key, result, ttlSeconds);
    
    logger.debug(`Memoized result for key: ${key}`);
    return result;
  }

  // Batch operations
  async setMultiple<T>(entries: Array<{ key: string; value: T; ttlSeconds?: number }>): Promise<void> {
    const promises = entries.map(({ key, value, ttlSeconds = 300 }) => 
      this.set(key, value, ttlSeconds)
    );
    
    await Promise.all(promises);
    logger.debug(`Cache SET BATCH: ${entries.length} entries`);
  }

  async getMultiple<T>(keys: string[]): Promise<Map<string, T | null>> {
    const results = new Map<string, T | null>();
    
    const promises = keys.map(async (key) => {
      const value = await this.get<T>(key);
      results.set(key, value);
    });

    await Promise.all(promises);
    logger.debug(`Cache GET BATCH: ${keys.length} keys`);
    return results;
  }

  private cleanupExpiredEntries() {
    const now = Date.now();
    let expiredCount = 0;

    this.cache.forEach((entry, key) => {
      const entryAge = now - entry.timestamp.getTime();
      if (entryAge > entry.ttl) {
        this.cache.delete(key);
        expiredCount++;
      }
    });

    if (expiredCount > 0) {
      logger.info(`Cache cleanup: removed ${expiredCount} expired entries`);
    }
  }
}
