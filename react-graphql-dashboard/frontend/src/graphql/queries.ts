import { gql } from '@apollo/client';

// Fragment for user data to reduce duplication and improve caching
export const USER_FRAGMENT = gql`
  fragment UserFragment on User {
    id
    name
    email
    avatar
    role
    lastActive
    status
  }
`;

// Fragment for metrics to optimize dashboard queries
export const METRICS_FRAGMENT = gql`
  fragment MetricsFragment on Metrics {
    id
    value
    change
    changePercent
    period
    timestamp
  }
`;

// Optimized dashboard query with field selection and caching
export const GET_DASHBOARD_DATA = gql`
  query GetDashboardData($timeRange: TimeRange!, $metrics: [MetricType!]!) {
    dashboardMetrics(timeRange: $timeRange, metrics: $metrics) @cached(ttl: 300) {
      ...MetricsFragment
      trend {
        timestamp
        value
      }
    }
    
    recentActivity(limit: 10) @cached(ttl: 60) {
      id
      type
      message
      timestamp
      user {
        ...UserFragment
      }
    }
    
    systemStatus @cached(ttl: 30) {
      cpu
      memory
      storage
      network
      healthy
    }
  }
  ${METRICS_FRAGMENT}
  ${USER_FRAGMENT}
`;

// Optimized user list query with pagination and filtering
export const GET_USERS = gql`
  query GetUsers(
    $first: Int = 20
    $after: String
    $filters: UserFilters
    $sortBy: UserSortField = CREATED_AT
    $sortOrder: SortOrder = DESC
  ) {
    users(
      first: $first
      after: $after
      filters: $filters
      sortBy: $sortBy
      sortOrder: $sortOrder
    ) @cached(ttl: 120) {
      edges {
        node {
          ...UserFragment
          createdAt
          updatedAt
          statistics {
            loginCount
            lastLoginAt
            activeProjects
          }
        }
        cursor
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
      totalCount
    }
  }
  ${USER_FRAGMENT}
`;

// Real-time subscription for live updates (not cached)
export const DASHBOARD_UPDATES_SUBSCRIPTION = gql`
  subscription DashboardUpdates {
    metricsUpdated {
      ...MetricsFragment
    }
    
    activityAdded {
      id
      type
      message
      timestamp
      user {
        ...UserFragment
      }
    }
    
    systemStatusChanged {
      cpu
      memory
      storage
      network
      healthy
    }
  }
  ${METRICS_FRAGMENT}
  ${USER_FRAGMENT}
`;

// Efficient search query with debouncing support
export const SEARCH_QUERY = gql`
  query Search(
    $query: String!
    $types: [SearchType!] = [USER, PROJECT, ACTIVITY]
    $limit: Int = 10
  ) {
    search(query: $query, types: $types, limit: $limit) @cached(ttl: 180) {
      ... on User {
        ...UserFragment
        __typename
      }
      ... on Project {
        id
        name
        description
        status
        owner {
          ...UserFragment
        }
        __typename
      }
      ... on Activity {
        id
        type
        message
        timestamp
        user {
          ...UserFragment
        }
        __typename
      }
    }
  }
  ${USER_FRAGMENT}
`;

// Mutation with optimistic updates
export const UPDATE_USER = gql`
  mutation UpdateUser($id: ID!, $input: UpdateUserInput!) {
    updateUser(id: $id, input: $input) {
      success
      user {
        ...UserFragment
        updatedAt
      }
      errors {
        field
        message
      }
    }
  }
  ${USER_FRAGMENT}
`;

// Batch mutation for bulk operations
export const BULK_UPDATE_USERS = gql`
  mutation BulkUpdateUsers($operations: [BulkUserOperation!]!) {
    bulkUpdateUsers(operations: $operations) {
      successful {
        ...UserFragment
      }
      failed {
        id
        error
      }
      totalProcessed
      successCount
      errorCount
    }
  }
  ${USER_FRAGMENT}
`;

// Performance analytics query for monitoring
export const GET_PERFORMANCE_METRICS = gql`
  query GetPerformanceMetrics($timeRange: TimeRange!) {
    performanceMetrics(timeRange: $timeRange) @cached(ttl: 60) {
      queryLatency {
        average
        p95
        p99
        trend {
          timestamp
          value
        }
      }
      
      cacheHitRate {
        overall
        byQuery {
          queryName
          hitRate
          missCount
        }
      }
      
      errorRate {
        rate
        count
        byType {
          type
          count
          percentage
        }
      }
      
      throughput {
        requestsPerSecond
        queriesPerSecond
        trend {
          timestamp
          value
        }
      }
    }
  }
`;

// Query for critical path optimization
export const GET_CRITICAL_DATA = gql`
  query GetCriticalData {
    # High priority data loaded first
    criticalMetrics @cached(ttl: 30) {
      systemHealth
      errorCount
      alertCount
    }
    
    # Secondary data loaded after critical path
    secondaryData @defer {
      userCount
      projectCount
      recentActivities(limit: 5) {
        id
        type
        timestamp
      }
    }
  }
`;