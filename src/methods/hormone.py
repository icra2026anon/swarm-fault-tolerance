"""
Hormone-based fault detection.
"""

from typing import Dict, Any, Tuple
import numpy as np
from ..core.simulator import BaseSimulator, SimulationConfig
from ..core.agent import BaseAgent, FaultType


class HormoneAgent(BaseAgent):
    """Agent with hormone capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suspicion_hormone = 0.0
        self.suspicion_emission = 0.0
        
    def update_suspicion_emission(self, threshold: float) -> None:
        """Update suspicion based on stress."""
        if self.stress_level > threshold * 0.5:
            self.suspicion_emission = self.stress_level * 0.1
        else:
            self.suspicion_emission = 0
            
    def decay_suspicion(self, rate: float) -> None:
        """Decay hormone over time."""
        self.suspicion_hormone *= rate


class HormoneSimulator(BaseSimulator):
    """Hormone-based detection simulator."""
    
    def __init__(self, config: SimulationConfig, enable_quarantine: bool = True):
        super().__init__(config)
        self.enable_quarantine = enable_quarantine
        self.quarantine_times = {}
        
    def _create_agent(self, agent_id: int, position: Tuple[float, float], 
                      fault_type: FaultType, arena_size: float) -> BaseAgent:
        """Create hormone agent."""
        return HormoneAgent(
            agent_id=agent_id,
            position=position,
            fault_type=fault_type,
            arena_size=arena_size,
            packet_loss_rate=self.config.packet_loss_rate,
            sensor_noise_level=self.config.sensor_noise_level
        )
        
    def get_adaptive_parameters(self, density: float) -> Dict[str, Any]:
        """Get parameters based on density."""
        if self.config.scaling_mode.value == 'fixed_arena':
            if density > 0.01:
                return {
                    'stress_threshold': 60,
                    'quarantine_delay': 0.8,
                    'damage_rate': 0.006,
                    'recovery_rate': 0.008,
                    'suspicion_amplifier': 1.0,
                    'suspicion_decay_rate': 0.95,
                    'suspicion_range_factor': 1.0
                }
            elif density > 0.007:
                return {
                    'stress_threshold': 80,
                    'quarantine_delay': 1.0,
                    'damage_rate': 0.005,
                    'recovery_rate': 0.006,
                    'suspicion_amplifier': 1.2,
                    'suspicion_decay_rate': 0.94,
                    'suspicion_range_factor': 1.1
                }
            elif density > 0.005:
                return {
                    'stress_threshold': 100,
                    'quarantine_delay': 1.2,
                    'damage_rate': 0.004,
                    'recovery_rate': 0.005,
                    'suspicion_amplifier': 1.5,
                    'suspicion_decay_rate': 0.93,
                    'suspicion_range_factor': 1.2
                }
            else:
                return {
                    'stress_threshold': 120,
                    'quarantine_delay': 1.5,
                    'damage_rate': 0.003,
                    'recovery_rate': 0.005,
                    'suspicion_amplifier': 2.0,
                    'suspicion_decay_rate': 0.90,
                    'suspicion_range_factor': 1.3
                }
        else:
            return {
                'stress_threshold': 80,
                'quarantine_delay': 1.2,
                'damage_rate': 0.004,
                'recovery_rate': 0.005,
                'suspicion_amplifier': 1.3,
                'suspicion_decay_rate': 0.93,
                'suspicion_range_factor': 1.1
            }
            
    def initialize_fault_detection(self, density: float) -> None:
        """Initialize detection."""
        self.quarantine_times.clear()
        
    async def update_fault_detection(self, elapsed: float, density: float) -> None:
        """Update hormone-based detection."""
        if not self.enable_quarantine:
            return
            
        params = self.get_adaptive_parameters(density)
        
        if elapsed <= params['quarantine_delay']:
            return
            
        # Update emissions
        for agent in self.agents:
            if agent.is_quarantined:
                continue
                
            agent.update_suspicion_emission(params['stress_threshold'])
            
            persistence = agent.get_stress_persistence()
            if persistence > 0.5:
                agent.suspicion_emission *= (1 + persistence)
                
            agent.decay_suspicion(params['suspicion_decay_rate'])
            
        # Accumulate suspicion
        for i, agent in enumerate(self.agents):
            if agent.is_quarantined:
                continue
                
            suspicion = 0
            contributors = 0
            range_factor = self.COMMUNICATION_RANGE * params['suspicion_range_factor']
            
            for j, other in enumerate(self.agents):
                if i == j or other.is_quarantined:
                    continue
                    
                dist = agent.distance_to(other)
                
                if dist <= range_factor and other.stress_level > params['stress_threshold'] * 0.7:
                    sensed = agent.sense_neighbor_stress(other.suspicion_emission)
                    
                    if sensed > 0:
                        decay_dist = 15 / params['suspicion_range_factor']
                        contrib = sensed * np.exp(-dist / decay_dist) * params['suspicion_amplifier']
                        
                        if self.config.scaling_mode.value == 'fixed_arena' and density > 0.005:
                            contrib *= min(1 + (density - 0.005) * 10, 1.3)
                            
                        suspicion += contrib
                        contributors += 1
                        
            if density < 0.005 and contributors > 0:
                suspicion *= min(1 + (0.005 - density) * 100, 1.5)
                
            agent.suspicion_hormone += suspicion
            
        # Check quarantine
        for agent in self.agents:
            if agent.is_quarantined or not agent.is_faulty:
                continue
                
            threshold = params['stress_threshold'] * 1.5
            
            if self.config.sensor_noise_level > 0:
                threshold *= (1 + self.config.sensor_noise_level)
                
            if agent.fault_type.value == 'severe':
                threshold *= 0.8
            elif agent.fault_type.value == 'moderate':
                threshold *= 0.9
                
            if density < 0.005:
                threshold *= 0.7 + (0.3 * density / 0.005)
                
            if agent.get_stress_persistence() > 0.7:
                threshold *= 0.85
                
            if agent.suspicion_hormone > threshold:
                agent.is_quarantined = True
                agent.quarantine_time = elapsed
                self.quarantine_times[agent.agent_id] = elapsed
                
    def _calculate_response_time(self) -> float:
        """Get average response time."""
        if self.quarantine_times:
            return np.mean(list(self.quarantine_times.values()))
        return self.config.run_time