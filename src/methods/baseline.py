"""
Baseline method - no fault detection.
"""

from typing import Dict, Any
from ..core.simulator import BaseSimulator


class BaselineSimulator(BaseSimulator):
    """No fault detection baseline."""
    
    def get_adaptive_parameters(self, density: float) -> Dict[str, Any]:
        """Get damage/recovery rates based on density."""
        if self.config.scaling_mode.value == 'fixed_arena':
            if density > 0.01:
                return {'damage_rate': 0.006, 'recovery_rate': 0.008}
            elif density > 0.007:
                return {'damage_rate': 0.005, 'recovery_rate': 0.006}
            elif density > 0.005:
                return {'damage_rate': 0.004, 'recovery_rate': 0.005}
            else:
                return {'damage_rate': 0.003, 'recovery_rate': 0.005}
        else:
            return {'damage_rate': 0.004, 'recovery_rate': 0.005}
            
    def initialize_fault_detection(self, density: float) -> None:
        """No detection needed."""
        pass
        
    async def update_fault_detection(self, elapsed: float, density: float) -> None:
        """No detection updates."""
        pass
        
    def _calculate_response_time(self) -> float:
        """No response time."""
        return self.config.run_time