"""
Voting-based fault detection.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from ..core.simulator import BaseSimulator, SimulationConfig
from ..core.agent import BaseAgent, FaultType


class VotingAgent(BaseAgent):
    """Agent with voting capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.votes_received = {}
        self.voting_history = {}
        self.suspicion_score = 0.0
        
    def cast_vote(self, target_id: int, target_stress: float, distance: float,
                  current_time: float, params: Dict[str, Any]) -> Tuple[Optional[float], float]:
        """Vote on target agent."""
        if distance > self.arena_size / 3:
            return None, 0
            
        adjusted_stress = max(0, target_stress - params.get('background_stress_offset', 0))
        threshold = params['stress_threshold_base']
        
        distance_factor = np.exp(-distance / 30)
        perceived_stress = self.sense_neighbor_stress(adjusted_stress)
        
        confidence = distance_factor * params['confidence_multiplier']
        if self.sensor_noise_level > 0:
            confidence *= (1 - self.sensor_noise_level * 0.3)
            
        # Simple voting logic with severity bonus
        severity_mult = 1.0
        if target_stress > threshold * 2.0:
            severity_mult = params.get('severe_bonus', 1.3)
        elif target_stress > threshold * 1.5:
            severity_mult = params.get('moderate_bonus', 1.15)

        if perceived_stress > threshold * 1.3:
            vote = 1.0 * severity_mult
        elif perceived_stress > threshold:
            vote = 0.7 * severity_mult
        elif perceived_stress > threshold * 0.8:
            vote = 0.4 * severity_mult
        else:
            vote = 0.0

        vote = min(vote, 1.0)

        if target_id not in self.voting_history:
            self.voting_history[target_id] = []
        self.voting_history[target_id].append((vote, confidence, current_time))

        # Keep only recent
        self.voting_history[target_id] = [
            v for v in self.voting_history[target_id] if current_time - v[2] < 5.0
        ]

        return vote, min(confidence, 1.0)

    def receive_vote(self, voter_id: int, vote: float, confidence: float) -> None:
        """Get vote from another agent."""
        if self.can_communicate():
            self.votes_received[voter_id] = (vote, confidence)

    def clear_votes(self) -> None:
        """Clear votes for new round."""
        self.votes_received.clear()


class VotingSimulator(BaseSimulator):
    """Voting-based detection simulator."""

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self.voting_history = {}
        self.last_vote_time = -100  # Start with negative to ensure first vote happens
        self.vote_interval = 10

    def _create_agent(self, agent_id: int, position: Tuple[float, float],
                      fault_type: FaultType, arena_size: float) -> BaseAgent:
        """Create voting agent."""
        return VotingAgent(
            agent_id=agent_id,
            position=position,
            fault_type=fault_type,
            arena_size=arena_size,
            packet_loss_rate=self.config.packet_loss_rate,
            sensor_noise_level=self.config.sensor_noise_level
        )

    def get_adaptive_parameters(self, density: float) -> Dict[str, Any]:
        """Get voting parameters."""
        # Get base damage/recovery parameters
        if self.config.scaling_mode.value == 'fixed_arena':
            if density > 0.01:
                base_params = {'damage_rate': 0.006, 'recovery_rate': 0.008}
            elif density > 0.007:
                base_params = {'damage_rate': 0.005, 'recovery_rate': 0.006}
            elif density > 0.005:
                base_params = {'damage_rate': 0.004, 'recovery_rate': 0.005}
            else:
                base_params = {'damage_rate': 0.003, 'recovery_rate': 0.005}
        else:
            base_params = {'damage_rate': 0.004, 'recovery_rate': 0.005}

        if density >= 0.01:
            voting_params = {
                'vote_weight_threshold': 0.6,
                'min_voters_required': 4,
                'confidence_multiplier': 0.7,
                'stress_threshold_base': 72,
                'suspicion_threshold': 4.0,
                'suspicion_decay': 0.91,
                'background_stress_offset': 15,
                'quarantine_confidence_threshold': 0.5,
                'severe_bonus': 1.3,
                'moderate_bonus': 1.15
            }
        elif density > 0.007:
            voting_params = {
                'vote_weight_threshold': 0.7,
                'min_voters_required': 5,
                'confidence_multiplier': 0.6,
                'stress_threshold_base': 100,
                'suspicion_threshold': 12.0,
                'suspicion_decay': 0.85,
                'background_stress_offset': 25,
                'quarantine_confidence_threshold': 0.65
            }
        elif density > 0.005:
            voting_params = {
                'vote_weight_threshold': 0.65,
                'min_voters_required': 4,
                'confidence_multiplier': 0.7,
                'stress_threshold_base': 85,
                'suspicion_threshold': 10.0,
                'suspicion_decay': 0.87,
                'background_stress_offset': 15,
                'quarantine_confidence_threshold': 0.6
            }
        else:
            # Sparse networks need lower thresholds due to fewer neighbors
            # and less frequent interactions (This part is very very difficult to tune)
            voting_params = {
                'vote_weight_threshold': 0.5,  # Lower for sparse
                'min_voters_required': 2,      # Lower requirement
                'confidence_multiplier': 0.8,
                'stress_threshold_base': 60,   # Lower threshold
                'suspicion_threshold': 2.5,    # Much lower (was 4.0)
                'suspicion_decay': 0.98,       # Much slower decay (was 0.95)
                'background_stress_offset': 5,
                'quarantine_confidence_threshold': 0.4,
                'severe_bonus': 1.3,
                'moderate_bonus': 1.15
            }

        return {**base_params, **voting_params}

    def initialize_fault_detection(self, density: float) -> None:
        """Initialize voting."""
        self.voting_history.clear()
        for agent in self.agents:
            self.voting_history[agent.agent_id] = {
                'suspicion_score': 0.0,
                'quarantine_time': None
            }

    async def update_fault_detection(self, elapsed: float, density: float) -> None:
        """Update voting detection."""
        params = self.get_adaptive_parameters(density)

        if elapsed <= 1.0:  # Start earlier
            return

        current_step = int(elapsed * 10)
        if current_step - self.last_vote_time < self.vote_interval:
            return

        self.last_vote_time = current_step

        # Clear votes
        for agent in self.agents:
            agent.clear_votes()

        # Collect votes
        for voter in self.agents:
            if voter.is_quarantined:
                continue

            for target in self.agents:
                if voter.agent_id == target.agent_id or target.is_quarantined:
                    continue

                dist = voter.distance_to(target)

                if dist <= self.COMMUNICATION_RANGE * 1.2:
                    vote, confidence = voter.cast_vote(
                        target.agent_id, target.stress_level, dist, elapsed, params
                    )

                    if vote is not None:
                        target.receive_vote(voter.agent_id, vote, confidence)

        # Make decisions
        decisions_made = 0
        high_suspicion = 0
        for agent in self.agents:
            if agent.is_quarantined:
                continue

            self.voting_history[agent.agent_id]['suspicion_score'] *= params['suspicion_decay']

            if self.voting_history[agent.agent_id]['suspicion_score'] > 1.0:
                high_suspicion += 1

            if len(agent.votes_received) < params['min_voters_required']:
                continue

            # Calculate consensus
            weighted_sum = 0
            weight_total = 0
            max_vote = 0
            strong_votes = 0

            for voter_id, (vote, confidence) in agent.votes_received.items():
                weighted_sum += vote * confidence
                weight_total += confidence
                max_vote = max(max_vote, vote)

                if vote > 0.6 and confidence > params.get('quarantine_confidence_threshold', 0.5):
                    strong_votes += 1

            if weight_total > 0:
                consensus = weighted_sum / weight_total
                avg_confidence = weight_total / len(agent.votes_received)

                # For sparse networks (density 0.005), be less strict
                if density < 0.005:
                    min_strong = 1
                    min_confidence = 0.35
                    min_consensus = 0.45
                else:
                    min_strong = 1
                    min_confidence = 0.42
                    min_consensus = 0.52

                if strong_votes < min_strong or avg_confidence < min_confidence or consensus < min_consensus:
                    continue

                if consensus > params['vote_weight_threshold']:
                    increment = (consensus - params['vote_weight_threshold']) * 2
                    self.voting_history[agent.agent_id]['suspicion_score'] += increment

                if self.voting_history[agent.agent_id]['suspicion_score'] >= params['suspicion_threshold']:
                    if agent.stress_level > params['stress_threshold_base'] * 0.5 or consensus > 0.7:
                        agent.is_quarantined = True
                        agent.quarantine_time = elapsed
                        self.voting_history[agent.agent_id]['quarantine_time'] = elapsed
                        
    def _calculate_response_time(self) -> float:
        """Get average response time."""
        times = []
        for history in self.voting_history.values():
            if history.get('quarantine_time'):
                times.append(history['quarantine_time'])
                
        if times:
            return np.mean(times)
        return self.config.run_time