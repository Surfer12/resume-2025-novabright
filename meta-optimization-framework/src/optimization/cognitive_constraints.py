"""
Cognitive Constraints Implementation

Implements cognitive constraints for the optimization framework, including
working memory limitations, attention capacity, processing speed constraints,
and cognitive load modeling.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of cognitive constraints."""
    WORKING_MEMORY = "working_memory"
    ATTENTION_CAPACITY = "attention_capacity"
    PROCESSING_SPEED = "processing_speed"
    COGNITIVE_LOAD = "cognitive_load"
    INTERFERENCE = "interference"
    FATIGUE = "fatigue"
    LEARNING_RATE = "learning_rate"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    constraint_type: ConstraintType
    severity: float  # 0.0 to 1.0
    description: str
    suggested_action: str
    timestamp: Optional[float] = None
    
    def __str__(self) -> str:
        return f"{self.constraint_type.value}: {self.description} (severity: {self.severity:.3f})"


@dataclass
class CognitiveCapacity:
    """Represents cognitive capacity limits."""
    working_memory_slots: int = 7  # Miller's 7±2
    attention_resources: float = 1.0
    processing_bandwidth: float = 1.0
    max_cognitive_load: float = 1.0
    interference_threshold: float = 0.7
    fatigue_accumulation_rate: float = 0.01


class WorkingMemoryConstraint:
    """
    Working memory constraint based on cognitive psychology research.
    
    Implements capacity limitations, decay, and interference effects
    in working memory processing.
    """
    
    def __init__(self, 
                 capacity: int = 7,
                 decay_rate: float = 0.1,
                 interference_factor: float = 0.2):
        """
        Initialize working memory constraint.
        
        Args:
            capacity: Maximum number of items in working memory
            decay_rate: Rate of memory decay per time step
            interference_factor: Factor for interference between items
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.interference_factor = interference_factor
        self.current_load = 0
        self.memory_items = []
        
    def check_constraint(self, 
                        memory_states: List[torch.Tensor],
                        sequence_length: int) -> List[ConstraintViolation]:
        """
        Check working memory constraints.
        
        Args:
            memory_states: List of memory state tensors
            sequence_length: Length of processing sequence
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Check capacity violations
        for i, memory_state in enumerate(memory_states):
            # Estimate number of active memory items
            active_items = self._count_active_items(memory_state)
            
            if active_items > self.capacity:
                severity = min(1.0, (active_items - self.capacity) / self.capacity)
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.WORKING_MEMORY,
                    severity=severity,
                    description=f"Working memory overload: {active_items} items > {self.capacity} capacity",
                    suggested_action="Reduce memory load or increase capacity"
                ))
        
        # Check for excessive memory decay
        if len(memory_states) > 1:
            decay_violations = self._check_decay_violations(memory_states)
            violations.extend(decay_violations)
        
        return violations
    
    def _count_active_items(self, memory_state: torch.Tensor) -> int:
        """Count number of active items in memory state."""
        # Use activation threshold to determine active items
        activation_threshold = 0.1
        active_mask = torch.norm(memory_state, dim=-1) > activation_threshold
        return int(torch.sum(active_mask).item())
    
    def _check_decay_violations(self, memory_states: List[torch.Tensor]) -> List[ConstraintViolation]:
        """Check for violations related to memory decay."""
        violations = []
        
        for i in range(1, len(memory_states)):
            prev_state = memory_states[i-1]
            curr_state = memory_states[i]
            
            # Compute decay rate
            state_diff = torch.norm(curr_state - prev_state)
            expected_decay = self.decay_rate * torch.norm(prev_state)
            
            if state_diff > expected_decay * 2:  # Excessive change
                severity = min(1.0, state_diff / (expected_decay * 2))
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.WORKING_MEMORY,
                    severity=severity,
                    description=f"Excessive memory state change: {state_diff:.3f}",
                    suggested_action="Reduce memory update rate or increase stability"
                ))
        
        return violations


class AttentionConstraint:
    """
    Attention capacity constraint based on attention research.
    
    Models limited attention resources, divided attention costs,
    and attention switching penalties.
    """
    
    def __init__(self, 
                 total_capacity: float = 1.0,
                 switching_cost: float = 0.1,
                 division_penalty: float = 0.2):
        """
        Initialize attention constraint.
        
        Args:
            total_capacity: Total attention capacity
            switching_cost: Cost of switching attention
            division_penalty: Penalty for dividing attention
        """
        self.total_capacity = total_capacity
        self.switching_cost = switching_cost
        self.division_penalty = division_penalty
        self.current_allocation = {}
        self.previous_allocation = {}
        
    def check_constraint(self, 
                        attention_weights: torch.Tensor,
                        attention_control: torch.Tensor) -> List[ConstraintViolation]:
        """
        Check attention constraints.
        
        Args:
            attention_weights: Attention weight distributions
            attention_control: Attention control signals
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Check total capacity violations
        total_attention = torch.sum(attention_weights, dim=-1)
        capacity_violations = total_attention > self.total_capacity
        
        if torch.any(capacity_violations):
            max_violation = torch.max(total_attention).item()
            severity = min(1.0, (max_violation - self.total_capacity) / self.total_capacity)
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.ATTENTION_CAPACITY,
                severity=severity,
                description=f"Attention overallocation: {max_violation:.3f} > {self.total_capacity}",
                suggested_action="Reduce attention demands or increase capacity"
            ))
        
        # Check attention switching costs
        switching_violations = self._check_switching_violations(attention_weights)
        violations.extend(switching_violations)
        
        # Check divided attention penalties
        division_violations = self._check_division_violations(attention_weights)
        violations.extend(division_violations)
        
        return violations
    
    def _check_switching_violations(self, attention_weights: torch.Tensor) -> List[ConstraintViolation]:
        """Check for excessive attention switching."""
        violations = []
        
        # Compute attention switching between time steps
        if attention_weights.shape[1] > 1:  # Multiple time steps
            attention_diff = torch.diff(attention_weights, dim=1)
            switching_magnitude = torch.norm(attention_diff, dim=-1)
            
            excessive_switching = switching_magnitude > self.switching_cost * 5
            
            if torch.any(excessive_switching):
                max_switching = torch.max(switching_magnitude).item()
                severity = min(1.0, max_switching / (self.switching_cost * 5))
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.ATTENTION_CAPACITY,
                    severity=severity,
                    description=f"Excessive attention switching: {max_switching:.3f}",
                    suggested_action="Reduce attention switching frequency"
                ))
        
        return violations
    
    def _check_division_violations(self, attention_weights: torch.Tensor) -> List[ConstraintViolation]:
        """Check for excessive attention division."""
        violations = []
        
        # Compute attention entropy as measure of division
        attention_probs = F.softmax(attention_weights, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
        
        # High entropy indicates divided attention
        max_entropy = np.log(attention_weights.shape[-1])  # Maximum possible entropy
        division_ratio = entropy / max_entropy
        
        excessive_division = division_ratio > (1 - self.division_penalty)
        
        if torch.any(excessive_division):
            max_division = torch.max(division_ratio).item()
            severity = min(1.0, (max_division - (1 - self.division_penalty)) / self.division_penalty)
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.ATTENTION_CAPACITY,
                severity=severity,
                description=f"Excessive attention division: {max_division:.3f}",
                suggested_action="Focus attention or increase division tolerance"
            ))
        
        return violations


class ProcessingSpeedConstraint:
    """
    Processing speed constraint based on cognitive timing research.
    
    Models reaction time limitations, speed-accuracy tradeoffs,
    and processing bottlenecks.
    """
    
    def __init__(self, 
                 max_processing_time: float = 1.0,
                 speed_accuracy_factor: float = 0.5,
                 bottleneck_threshold: float = 0.8):
        """
        Initialize processing speed constraint.
        
        Args:
            max_processing_time: Maximum allowed processing time
            speed_accuracy_factor: Factor for speed-accuracy tradeoff
            bottleneck_threshold: Threshold for detecting bottlenecks
        """
        self.max_processing_time = max_processing_time
        self.speed_accuracy_factor = speed_accuracy_factor
        self.bottleneck_threshold = bottleneck_threshold
        
    def check_constraint(self, 
                        processing_times: List[float],
                        accuracy_scores: List[float]) -> List[ConstraintViolation]:
        """
        Check processing speed constraints.
        
        Args:
            processing_times: List of processing times
            accuracy_scores: List of accuracy scores
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Check processing time violations
        for i, proc_time in enumerate(processing_times):
            if proc_time > self.max_processing_time:
                severity = min(1.0, (proc_time - self.max_processing_time) / self.max_processing_time)
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.PROCESSING_SPEED,
                    severity=severity,
                    description=f"Processing time exceeded: {proc_time:.3f}s > {self.max_processing_time}s",
                    suggested_action="Optimize processing or increase time limit"
                ))
        
        # Check speed-accuracy tradeoff violations
        if len(processing_times) == len(accuracy_scores):
            tradeoff_violations = self._check_speed_accuracy_tradeoff(processing_times, accuracy_scores)
            violations.extend(tradeoff_violations)
        
        return violations
    
    def _check_speed_accuracy_tradeoff(self, 
                                      processing_times: List[float],
                                      accuracy_scores: List[float]) -> List[ConstraintViolation]:
        """Check for violations of speed-accuracy tradeoff."""
        violations = []
        
        for i, (time, accuracy) in enumerate(zip(processing_times, accuracy_scores)):
            # Expected accuracy based on processing time
            expected_accuracy = min(1.0, time * self.speed_accuracy_factor)
            
            if accuracy < expected_accuracy * 0.8:  # Significant underperformance
                severity = (expected_accuracy - accuracy) / expected_accuracy
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.PROCESSING_SPEED,
                    severity=severity,
                    description=f"Poor speed-accuracy tradeoff: {accuracy:.3f} < {expected_accuracy:.3f}",
                    suggested_action="Balance processing speed and accuracy"
                ))
        
        return violations


class CognitiveLoadConstraint:
    """
    Cognitive load constraint based on cognitive load theory.
    
    Models intrinsic, extraneous, and germane cognitive load
    and their effects on performance.
    """
    
    def __init__(self, 
                 max_total_load: float = 1.0,
                 intrinsic_weight: float = 0.4,
                 extraneous_weight: float = 0.3,
                 germane_weight: float = 0.3):
        """
        Initialize cognitive load constraint.
        
        Args:
            max_total_load: Maximum total cognitive load
            intrinsic_weight: Weight for intrinsic load
            extraneous_weight: Weight for extraneous load
            germane_weight: Weight for germane load
        """
        self.max_total_load = max_total_load
        self.intrinsic_weight = intrinsic_weight
        self.extraneous_weight = extraneous_weight
        self.germane_weight = germane_weight
        
    def check_constraint(self, 
                        task_complexity: float,
                        processing_efficiency: float,
                        learning_engagement: float) -> List[ConstraintViolation]:
        """
        Check cognitive load constraints.
        
        Args:
            task_complexity: Intrinsic task complexity
            processing_efficiency: Processing efficiency (inverse of extraneous load)
            learning_engagement: Learning engagement (germane load)
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Compute load components
        intrinsic_load = task_complexity * self.intrinsic_weight
        extraneous_load = (1 - processing_efficiency) * self.extraneous_weight
        germane_load = learning_engagement * self.germane_weight
        
        total_load = intrinsic_load + extraneous_load + germane_load
        
        if total_load > self.max_total_load:
            severity = min(1.0, (total_load - self.max_total_load) / self.max_total_load)
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.COGNITIVE_LOAD,
                severity=severity,
                description=f"Cognitive overload: {total_load:.3f} > {self.max_total_load}",
                suggested_action="Reduce task complexity or improve efficiency"
            ))
        
        # Check for excessive extraneous load
        if extraneous_load > self.max_total_load * 0.5:
            severity = extraneous_load / (self.max_total_load * 0.5) - 1
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.COGNITIVE_LOAD,
                severity=severity,
                description=f"Excessive extraneous load: {extraneous_load:.3f}",
                suggested_action="Improve processing efficiency"
            ))
        
        return violations


class CognitiveConstraints:
    """
    Main cognitive constraints system.
    
    Integrates multiple cognitive constraints and provides unified
    constraint checking and violation reporting.
    """
    
    def __init__(self, capacity: CognitiveCapacity = None):
        """
        Initialize cognitive constraints system.
        
        Args:
            capacity: Cognitive capacity configuration
        """
        self.capacity = capacity or CognitiveCapacity()
        
        # Initialize constraint modules
        self.working_memory = WorkingMemoryConstraint(
            capacity=self.capacity.working_memory_slots,
            decay_rate=0.1,
            interference_factor=0.2
        )
        
        self.attention = AttentionConstraint(
            total_capacity=self.capacity.attention_resources,
            switching_cost=0.1,
            division_penalty=0.2
        )
        
        self.processing_speed = ProcessingSpeedConstraint(
            max_processing_time=1.0,
            speed_accuracy_factor=0.5,
            bottleneck_threshold=0.8
        )
        
        self.cognitive_load = CognitiveLoadConstraint(
            max_total_load=self.capacity.max_cognitive_load,
            intrinsic_weight=0.4,
            extraneous_weight=0.3,
            germane_weight=0.3
        )
        
        # Violation history
        self.violation_history = []
        
        logger.info("Initialized CognitiveConstraints system")
    
    def check_all_constraints(self, 
                             system_state: Dict[str, Any]) -> List[ConstraintViolation]:
        """
        Check all cognitive constraints against system state.
        
        Args:
            system_state: Current system state with relevant metrics
            
        Returns:
            List of all constraint violations
        """
        all_violations = []
        
        # Working memory constraints
        if 'memory_states' in system_state:
            wm_violations = self.working_memory.check_constraint(
                system_state['memory_states'],
                system_state.get('sequence_length', 1)
            )
            all_violations.extend(wm_violations)
        
        # Attention constraints
        if 'attention_weights' in system_state and 'attention_control' in system_state:
            att_violations = self.attention.check_constraint(
                system_state['attention_weights'],
                system_state['attention_control']
            )
            all_violations.extend(att_violations)
        
        # Processing speed constraints
        if 'processing_times' in system_state and 'accuracy_scores' in system_state:
            speed_violations = self.processing_speed.check_constraint(
                system_state['processing_times'],
                system_state['accuracy_scores']
            )
            all_violations.extend(speed_violations)
        
        # Cognitive load constraints
        if all(key in system_state for key in ['task_complexity', 'processing_efficiency', 'learning_engagement']):
            load_violations = self.cognitive_load.check_constraint(
                system_state['task_complexity'],
                system_state['processing_efficiency'],
                system_state['learning_engagement']
            )
            all_violations.extend(load_violations)
        
        # Store violations in history
        self.violation_history.extend(all_violations)
        
        return all_violations
    
    def compute_constraint_penalty(self, violations: List[ConstraintViolation]) -> float:
        """
        Compute penalty score based on constraint violations.
        
        Args:
            violations: List of constraint violations
            
        Returns:
            Penalty score (0.0 = no violations, 1.0 = severe violations)
        """
        if not violations:
            return 0.0
        
        # Weighted penalty based on violation severity
        total_penalty = 0.0
        for violation in violations:
            # Weight by constraint type importance
            type_weight = {
                ConstraintType.WORKING_MEMORY: 0.3,
                ConstraintType.ATTENTION_CAPACITY: 0.25,
                ConstraintType.PROCESSING_SPEED: 0.2,
                ConstraintType.COGNITIVE_LOAD: 0.25
            }.get(violation.constraint_type, 0.1)
            
            total_penalty += violation.severity * type_weight
        
        # Normalize by number of violations
        avg_penalty = total_penalty / len(violations)
        
        return min(1.0, avg_penalty)
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of constraint system state."""
        recent_violations = self.violation_history[-100:]  # Last 100 violations
        
        violation_counts = {}
        for violation in recent_violations:
            constraint_type = violation.constraint_type.value
            violation_counts[constraint_type] = violation_counts.get(constraint_type, 0) + 1
        
        avg_severity = np.mean([v.severity for v in recent_violations]) if recent_violations else 0.0
        
        return {
            'total_violations': len(recent_violations),
            'violation_counts': violation_counts,
            'average_severity': avg_severity,
            'capacity_settings': {
                'working_memory_slots': self.capacity.working_memory_slots,
                'attention_resources': self.capacity.attention_resources,
                'max_cognitive_load': self.capacity.max_cognitive_load
            }
        }
    
    def adjust_capacity(self, capacity_updates: Dict[str, float]) -> None:
        """
        Adjust cognitive capacity parameters.
        
        Args:
            capacity_updates: Dictionary of capacity parameter updates
        """
        for param, value in capacity_updates.items():
            if hasattr(self.capacity, param):
                setattr(self.capacity, param, value)
                logger.info(f"Updated {param} to {value}")
        
        # Update constraint modules with new capacity
        self.working_memory.capacity = self.capacity.working_memory_slots
        self.attention.total_capacity = self.capacity.attention_resources
        self.cognitive_load.max_total_load = self.capacity.max_cognitive_load
    
    def reset_violation_history(self) -> None:
        """Reset violation history."""
        self.violation_history = []
        logger.info("Reset constraint violation history")