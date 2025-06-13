import { gql } from '@apollo/client';

// Fragment for consciousness metrics
export const CONSCIOUSNESS_METRICS_FRAGMENT = gql`
  fragment ConsciousnessMetricsFragment on ConsciousnessMetrics {
    accuracyImprovement
    cognitiveLoadReduction
    integrationLevel
    efficiencyGains
    biasAccuracy
    consciousnessLevel
    evolutionStage
    alpha
    lambda1
    lambda2
    beta
    timestamp
  }
`;

// Query for consciousness metrics with parameter optimization
export const GET_CONSCIOUSNESS_METRICS = gql`
  query GetConsciousnessMetrics(
    $alpha: Float!
    $lambda1: Float!
    $lambda2: Float!
    $beta: Float!
  ) {
    consciousnessMetrics(
      alpha: $alpha
      lambda1: $lambda1
      lambda2: $lambda2
      beta: $beta
    ) @cached(ttl: 60) {
      ...ConsciousnessMetricsFragment
      phaseSpaceData {
        x
        y
        z
        timestamp
      }
      neuralNetworkState {
        layers {
          neurons {
            id
            activation
            position {
              x
              y
              z
            }
            connections {
              targetId
              strength
              type
            }
          }
        }
        globalWorkspace {
          integration
          consciousness
          emergence
        }
      }
      algorithmStatus {
        grandUnified
        dynamicIntegration
        cognitiveRegularization
        biasModeling
        metaOptimization
      }
    }
  }
  ${CONSCIOUSNESS_METRICS_FRAGMENT}
`;

// Real-time subscription for consciousness updates
export const CONSCIOUSNESS_UPDATES_SUBSCRIPTION = gql`
  subscription ConsciousnessUpdates {
    consciousnessUpdated {
      ...ConsciousnessMetricsFragment
      updateType
      changedParameters
    }
    
    neuralNetworkUpdated {
      layerId
      neuronId
      activation
      newConnections {
        targetId
        strength
      }
    }
    
    algorithmStatusChanged {
      algorithm
      status
      performance
      timestamp
    }
  }
  ${CONSCIOUSNESS_METRICS_FRAGMENT}
`;

// Query for consciousness performance analytics
export const GET_CONSCIOUSNESS_PERFORMANCE = gql`
  query GetConsciousnessPerformance($timeRange: TimeRange!) {
    consciousnessPerformance(timeRange: $timeRange) @cached(ttl: 120) {
      optimizationHistory {
        timestamp
        accuracy
        efficiency
        consciousness
        stage
      }
      
      convergenceMetrics {
        rate
        stability
        oscillation
        targetDistance
      }
      
      emergenceIndicators {
        complexity
        selfOrganization
        adaptation
        creativity
      }
      
      benchmarkComparison {
        baseline {
          latency
          accuracy
          efficiency
        }
        optimized {
          latency
          accuracy
          efficiency
        }
        improvement {
          latencyReduction
          accuracyGain
          efficiencyGain
        }
      }
    }
  }
`;

// Mutation for updating consciousness parameters
export const UPDATE_CONSCIOUSNESS_PARAMETERS = gql`
  mutation UpdateConsciousnessParameters(
    $alpha: Float!
    $lambda1: Float!
    $lambda2: Float!
    $beta: Float!
    $realTime: Boolean = true
  ) {
    updateConsciousnessParameters(
      alpha: $alpha
      lambda1: $lambda1
      lambda2: $lambda2
      beta: $beta
      realTime: $realTime
    ) {
      success
      metrics {
        ...ConsciousnessMetricsFragment
      }
      errors {
        field
        message
      }
      performance {
        updateTime
        convergenceTime
        stabilityScore
      }
    }
  }
  ${CONSCIOUSNESS_METRICS_FRAGMENT}
`;

// Query for consciousness insights and recommendations
export const GET_CONSCIOUSNESS_INSIGHTS = gql`
  query GetConsciousnessInsights {
    consciousnessInsights @cached(ttl: 300) {
      recommendations {
        parameter
        currentValue
        suggestedValue
        expectedImprovement
        reasoning
        confidence
      }
      
      patterns {
        type
        description
        frequency
        significance
        trend
      }
      
      emergentBehaviors {
        behavior
        strength
        stability
        conditions
        implications
      }
      
      performanceThresholds {
        consciousness
        accuracy
        efficiency
        emergence
      }
    }
  }
`;