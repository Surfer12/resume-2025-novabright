import { gql } from 'graphql-tag';

export const typeDefs = gql`
  # Custom scalars for better type safety
  scalar DateTime
  scalar JSON

  # Cache control directive for field-level caching
  directive @cached(
    ttl: Int = 300
    scope: CacheScope = PUBLIC
  ) on FIELD_DEFINITION

  # Defer directive for progressive loading
  directive @defer on FIELD

  enum CacheScope {
    PUBLIC
    PRIVATE
  }

  # Enums for better type safety and GraphQL optimization
  enum TimeRange {
    HOUR
    DAY
    WEEK
    MONTH
    YEAR
  }

  enum MetricType {
    CPU
    MEMORY
    STORAGE
    NETWORK
    REQUESTS
    ERRORS
    LATENCY
  }

  enum UserRole {
    ADMIN
    USER
    VIEWER
  }

  enum UserStatus {
    ACTIVE
    INACTIVE
    SUSPENDED
  }

  enum ActivityType {
    LOGIN
    LOGOUT
    CREATE
    UPDATE
    DELETE
    VIEW
    EXPORT
  }

  enum SortOrder {
    ASC
    DESC
  }

  enum UserSortField {
    NAME
    EMAIL
    CREATED_AT
    LAST_ACTIVE
  }

  enum SearchType {
    USER
    PROJECT
    ACTIVITY
  }

  # Core types with optimized field selection
  type User {
    id: ID!
    name: String!
    email: String!
    avatar: String
    role: UserRole!
    status: UserStatus!
    lastActive: DateTime
    createdAt: DateTime!
    updatedAt: DateTime!
    
    # Computed fields with caching
    statistics: UserStatistics @cached(ttl: 300)
  }

  type UserStatistics {
    loginCount: Int!
    lastLoginAt: DateTime
    activeProjects: Int!
    activityScore: Float!
  }

  type Metrics {
    id: ID!
    type: MetricType!
    value: Float!
    change: Float!
    changePercent: Float!
    period: String!
    timestamp: DateTime!
    
    # Trend data for charts (cached for better performance)
    trend: [TrendPoint!]! @cached(ttl: 180)
  }

  type TrendPoint {
    timestamp: DateTime!
    value: Float!
  }

  type SystemStatus {
    cpu: Float!
    memory: Float!
    storage: Float!
    network: Float!
    healthy: Boolean!
    lastUpdated: DateTime!
  }

  type Activity {
    id: ID!
    type: ActivityType!
    message: String!
    timestamp: DateTime!
    metadata: JSON
    
    # User relationship with DataLoader optimization
    user: User!
  }

  type Project {
    id: ID!
    name: String!
    description: String
    status: String!
    createdAt: DateTime!
    
    # Owner relationship with DataLoader
    owner: User!
  }

  # Performance metrics for monitoring the GraphQL API itself
  type PerformanceMetrics {
    queryLatency: LatencyMetrics!
    cacheHitRate: CacheMetrics!
    errorRate: ErrorMetrics!
    throughput: ThroughputMetrics!
  }

  type LatencyMetrics {
    average: Float!
    p50: Float!
    p95: Float!
    p99: Float!
    trend: [TrendPoint!]!
  }

  type CacheMetrics {
    overall: Float!
    byQuery: [QueryCacheMetrics!]!
  }

  type QueryCacheMetrics {
    queryName: String!
    hitRate: Float!
    missCount: Int!
  }

  type ErrorMetrics {
    rate: Float!
    count: Int!
    byType: [ErrorTypeMetrics!]!
  }

  type ErrorTypeMetrics {
    type: String!
    count: Int!
    percentage: Float!
  }

  type ThroughputMetrics {
    requestsPerSecond: Float!
    queriesPerSecond: Float!
    trend: [TrendPoint!]!
  }

  # Pagination types for efficient data loading
  type PageInfo {
    hasNextPage: Boolean!
    hasPreviousPage: Boolean!
    startCursor: String
    endCursor: String
  }

  type UserEdge {
    node: User!
    cursor: String!
  }

  type UserConnection {
    edges: [UserEdge!]!
    pageInfo: PageInfo!
    totalCount: Int!
  }

  # Input types for mutations and filters
  input UserFilters {
    role: UserRole
    status: UserStatus
    search: String
    createdAfter: DateTime
    createdBefore: DateTime
  }

  input UpdateUserInput {
    name: String
    email: String
    role: UserRole
    status: UserStatus
    avatar: String
  }

  input BulkUserOperation {
    id: ID!
    operation: String!
    data: JSON
  }

  # Search union type for polymorphic search results
  union SearchResult = User | Project | Activity

  # Mutation response types
  type UpdateUserResponse {
    success: Boolean!
    user: User
    errors: [FieldError!]!
  }

  type FieldError {
    field: String!
    message: String!
  }

  type BulkUpdateResponse {
    successful: [User!]!
    failed: [BulkOperationError!]!
    totalProcessed: Int!
    successCount: Int!
    errorCount: Int!
  }

  type BulkOperationError {
    id: ID!
    error: String!
  }

  # Root types with optimized resolvers
  type Query {
    # Dashboard data with aggressive caching
    dashboardMetrics(
      timeRange: TimeRange!
      metrics: [MetricType!]!
    ): [Metrics!]! @cached(ttl: 300)
    
    # Recent activity with moderate caching
    recentActivity(limit: Int = 10): [Activity!]! @cached(ttl: 60)
    
    # System status with short-term caching
    systemStatus: SystemStatus @cached(ttl: 30)
    
    # User queries with pagination
    users(
      first: Int = 20
      after: String
      filters: UserFilters
      sortBy: UserSortField = CREATED_AT
      sortOrder: SortOrder = DESC
    ): UserConnection! @cached(ttl: 120)
    
    user(id: ID!): User @cached(ttl: 300)
    
    # Search with moderate caching
    search(
      query: String!
      types: [SearchType!] = [USER, PROJECT, ACTIVITY]
      limit: Int = 10
    ): [SearchResult!]! @cached(ttl: 180)
    
    # Performance monitoring (shorter cache for real-time monitoring)
    performanceMetrics(timeRange: TimeRange!): PerformanceMetrics @cached(ttl: 60)
    
    # Critical data for initial page load
    criticalMetrics: CriticalMetrics @cached(ttl: 30)
    
    # Secondary data that can be deferred
    secondaryData: SecondaryData @defer
  }

  type CriticalMetrics {
    systemHealth: Float!
    errorCount: Int!
    alertCount: Int!
  }

  type SecondaryData {
    userCount: Int!
    projectCount: Int!
    recentActivities(limit: Int = 5): [Activity!]!
  }

  type Mutation {
    updateUser(id: ID!, input: UpdateUserInput!): UpdateUserResponse!
    bulkUpdateUsers(operations: [BulkUserOperation!]!): BulkUpdateResponse!
    
    # Cache invalidation mutations
    invalidateCache(keys: [String!]!): Boolean!
    warmCache(queries: [String!]!): Boolean!
  }

  type Subscription {
    # Real-time updates for live dashboard
    metricsUpdated: Metrics!
    activityAdded: Activity!
    systemStatusChanged: SystemStatus!
    
    # User-specific subscriptions
    userUpdated(userId: ID!): User!
  }
`;

export default typeDefs;