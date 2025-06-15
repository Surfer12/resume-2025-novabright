export const typeDefs = `
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

  # Consciousness-specific enums
  enum EvolutionStage {
    LINEAR
    RECURSIVE
    EMERGENT
  }

  enum ConsciousnessUpdateType {
    PARAMETER_CHANGE
    METRIC_UPDATE
    STAGE_TRANSITION
    OPTIMIZATION_COMPLETE
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
    statistics: UserStatistics
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
    trend: [TrendPoint!]!
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

  # Consciousness Types
  type ConsciousnessMetrics {
    accuracyImprovement: Float!
    cognitiveLoadReduction: Float!
    integrationLevel: Float!
    efficiencyGains: Float!
    biasAccuracy: Float!
    consciousnessLevel: Float!
    evolutionStage: EvolutionStage!
    alpha: Float!
    lambda1: Float!
    lambda2: Float!
    beta: Float!
    timestamp: DateTime!
    
    # Advanced visualization data
    phaseSpaceData: PhaseSpaceData!
    neuralNetworkState: NeuralNetworkState!
    algorithmStatus: AlgorithmStatus!
  }

  type PhaseSpaceData {
    x: [Float!]!
    y: [Float!]!
    z: [Float!]!
    timestamp: DateTime!
  }

  type NeuralNetworkState {
    layers: [NeuralLayer!]!
    globalWorkspace: GlobalWorkspace!
  }

  type NeuralLayer {
    id: ID!
    neurons: [Neuron!]!
  }

  type Neuron {
    id: ID!
    activation: Float!
    position: Position3D!
    connections: [NeuralConnection!]!
    consciousness: Float!
    stage: EvolutionStage!
  }

  type Position3D {
    x: Float!
    y: Float!
    z: Float!
  }

  type NeuralConnection {
    targetId: ID!
    strength: Float!
    type: String!
  }

  type GlobalWorkspace {
    integration: Float!
    consciousness: Float!
    emergence: Float!
  }

  type AlgorithmStatus {
    grandUnified: String!
    dynamicIntegration: String!
    cognitiveRegularization: String!
    biasModeling: String!
    metaOptimization: String!
  }

  type ConsciousnessPerformance {
    optimizationHistory: [OptimizationPoint!]!
    convergenceMetrics: ConvergenceMetrics!
    emergenceIndicators: EmergenceIndicators!
    benchmarkComparison: BenchmarkComparison!
  }

  type OptimizationPoint {
    timestamp: DateTime!
    accuracy: Float!
    efficiency: Float!
    consciousness: Float!
    stage: EvolutionStage!
  }

  type ConvergenceMetrics {
    rate: Float!
    stability: Float!
    oscillation: Float!
    targetDistance: Float!
  }

  type EmergenceIndicators {
    complexity: Float!
    selfOrganization: Float!
    adaptation: Float!
    creativity: Float!
  }

  type BenchmarkComparison {
    baseline: PerformanceStats!
    optimized: PerformanceStats!
    improvement: ImprovementStats!
  }

  type PerformanceStats {
    latency: Float!
    accuracy: Float!
    efficiency: Float!
  }

  type ImprovementStats {
    latencyReduction: Float!
    accuracyGain: Float!
    efficiencyGain: Float!
  }

  type ConsciousnessInsights {
    recommendations: [ParameterRecommendation!]!
    patterns: [ConsciousnessPattern!]!
    emergentBehaviors: [EmergentBehavior!]!
    performanceThresholds: PerformanceThresholds!
  }

  type ParameterRecommendation {
    parameter: String!
    currentValue: Float!
    suggestedValue: Float!
    expectedImprovement: Float!
    reasoning: String!
    confidence: Float!
  }

  type ConsciousnessPattern {
    type: String!
    description: String!
    frequency: Float!
    significance: Float!
    trend: String!
  }

  type EmergentBehavior {
    behavior: String!
    strength: Float!
    stability: Float!
    conditions: [String!]!
    implications: String!
  }

  type PerformanceThresholds {
    consciousness: Float!
    accuracy: Float!
    efficiency: Float!
    emergence: Float!
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

  # Consciousness mutation response types
  type UpdateConsciousnessParametersResponse {
    success: Boolean!
    metrics: ConsciousnessMetrics
    errors: [FieldError!]!
    performance: ParameterUpdatePerformance!
  }

  type ParameterUpdatePerformance {
    updateTime: Float!
    convergenceTime: Float!
    stabilityScore: Float!
  }

  # Root types with optimized resolvers
  type Query {
    # Dashboard data with aggressive caching
    dashboardMetrics(
      timeRange: TimeRange!
      metrics: [MetricType!]!
    ): [Metrics!]!
    
    # Recent activity with moderate caching
    recentActivity(limit: Int = 10): [Activity!]!
    
    # System status with short-term caching
    systemStatus: SystemStatus
    
    # User queries with pagination
    users(
      first: Int = 20
      after: String
      filters: UserFilters
      sortBy: UserSortField = CREATED_AT
      sortOrder: SortOrder = DESC
    ): UserConnection!
    
    user(id: ID!): User
    
    # Search with moderate caching
    search(
      query: String!
      types: [SearchType!] = [USER, PROJECT, ACTIVITY]
      limit: Int = 10
    ): [SearchResult!]!
    
    # Performance monitoring (shorter cache for real-time monitoring)
    performanceMetrics(timeRange: TimeRange!): PerformanceMetrics
    
    # Critical data for initial page load
    criticalMetrics: CriticalMetrics
    
    # Secondary data that can be deferred
    secondaryData: SecondaryData

    # Consciousness queries
    consciousnessMetrics(
      alpha: Float!
      lambda1: Float!
      lambda2: Float!
      beta: Float!
    ): ConsciousnessMetrics

    consciousnessPerformance(timeRange: TimeRange!): ConsciousnessPerformance

    consciousnessInsights: ConsciousnessInsights
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

    # Consciousness mutations
    updateConsciousnessParameters(
      alpha: Float!
      lambda1: Float!
      lambda2: Float!
      beta: Float!
      realTime: Boolean = true
    ): UpdateConsciousnessParametersResponse!
  }

  type Subscription {
    # Real-time updates for live dashboard
    metricsUpdated: Metrics
    activityAdded: Activity
    systemStatusChanged: SystemStatus
    
    # User-specific subscriptions
    userUpdated(userId: ID!): User
    
    # Consciousness subscriptions
    consciousnessUpdated: ConsciousnessMetrics
    neuralNetworkUpdated: NeuralNetworkState
    algorithmStatusChanged: AlgorithmStatus
  }
`;

export default typeDefs;