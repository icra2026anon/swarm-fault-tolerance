"""
Base agent class for swarm fault tolerance experiments.

This module defines the core agent implementation
for evaluating fault detection strategies in swarm
simulations, focusing on movement, fault modeling,
and performance tracking.
"""

from typing import Tuple, Optional, List
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum


class FaultType(Enum):
    """Enumerates possible agent fault types."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class AgentState:
    """Represents the current state of an agent."""
    position: np.ndarray
    velocity: np.ndarray
    task_rate: float
    stress_level: float
    is_quarantined: bool
    energy_consumed: float
    area_covered: float


class BaseAgent:
    """
    Implements core agent functionality for swarm simulations.

    Manages agent movement, fault modeling, performance degradation, energy consumption, and communication with noise and packet loss in a simulated environment.

    Attributes:
        agent_id: Unique agent identifier.
        position: 2D position in the arena (m).
        velocity: 2D velocity vector (m/s).
        fault_type: Type of fault, if any.
        is_faulty: Indicates if the agent has a fault.
        task_rate: Task completion efficiency [0, 1].
        stress_level: Accumulated stress from neighboring agents.
        is_quarantined: Indicates if the agent is quarantined.
        arena_size: Size of the square arena (m).
        packet_loss_rate: Probability of packet loss [0, 1].
        sensor_noise_level: Standard deviation multiplier for sensor noise.
    """

    # Energy constants - same values as in metrics.py
    ENERGY_IDLE = 0.1      # power when quarantined
    ENERGY_ACTIVE = 1.2    # power when active

    def __init__(
        self,
        agent_id: int,
        position: Tuple[float, float],
        fault_type: FaultType = FaultType.NONE,
        arena_size: float = 100.0,
        packet_loss_rate: float = 0.0,
        sensor_noise_level: float = 0.0
    ):
        """Initializes an agent with specified parameters."""
        self.agent_id = agent_id
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.random.uniform(-2, 2, size=2).astype(np.float64)
        self.fault_type = fault_type
        self.is_faulty = fault_type != FaultType.NONE
        self.arena_size = arena_size
        self.packet_loss_rate = packet_loss_rate
        self.sensor_noise_level = sensor_noise_level

        self._initialize_fault_parameters()
        self.is_quarantined = False
        self.quarantine_time: Optional[float] = None
        self.damage_taken = 0.0
        self.energy_consumed = 0.0
        self.current_coverage_area = 0.0  # renamed for clarity
        self.faulty_neighbors = 0
        self.time_near_faulty = 0.0
        self.time_away_from_faulty = 0.0
        self.stress_history: List[float] = []

    def _initialize_fault_parameters(self) -> None:
        """Sets initial performance parameters based on fault type."""
        # store the fault parameters in a dict
        fault_params = {
            FaultType.NONE: (1.0, 0.0),
            FaultType.MINOR: (0.7, 50.0),
            FaultType.MODERATE: (0.4, 120.0),
            FaultType.SEVERE: (0.2, 180.0)
        }
        # get the values from dict
        self.base_task_rate, self.stress_output = fault_params[self.fault_type]
        self.task_rate = self.base_task_rate
        self.stress_level = 0.0  # this is accumulated stress from neighbors

    def update_position(self, dt: float = 0.1) -> None:
        """Updates agent position with boundary reflection."""
        # update position based on velocity
        self.position += self.velocity * dt

        # handle boundaries
        margin = 5.0
        for i in range(2):
            # check if too close to left/bottom boundary
            if self.position[i] < margin:
                self.position[i] = margin
                self.velocity[i] = abs(self.velocity[i])
            # check if too close to right/top boundary
            elif self.position[i] > self.arena_size - margin:
                self.position[i] = self.arena_size - margin
                self.velocity[i] = -abs(self.velocity[i])

    def take_damage(self, damage: float) -> None:
        """Applies performance degradation to healthy agents."""
        # only healthy agents can take damage
        if not self.is_faulty:
            # update damage taken
            self.damage_taken = min(0.5, self.damage_taken + damage)
            # update task rate based on damage
            self.task_rate = max(0.5, 1.0 - self.damage_taken)

    def apply_recovery(self, recovery_rate: float = 0.005) -> None:
        """Recovers performance when away from faulty agents."""
        # check if agent can recover
        if not self.is_faulty and self.faulty_neighbors == 0:
            # calculate recovery amount
            recovery = recovery_rate * (1.0 - self.task_rate)
            # apply recovery
            self.task_rate = min(1.0, self.task_rate + recovery)
            self.damage_taken = max(0.0, self.damage_taken - recovery)

    def can_communicate(self) -> bool:
        """Simulates packet loss in communication."""
        # randomly determine if packet is lost
        return random.random() > self.packet_loss_rate

    def sense_neighbor_stress(self, actual_stress: float) -> float:
        """Returns sensed stress with added noise."""
        # check if communication works
        if not self.can_communicate():
            return 0.0

        # add noise to stress reading
        noise = np.random.normal(0, self.sensor_noise_level * actual_stress)
        sensed_stress = actual_stress + noise

        # clamp to reasonable range (0 to 2x actual)
        if actual_stress > 0:
            return np.clip(sensed_stress, 0.0, actual_stress * 2.0)
        else:
            return max(0.0, sensed_stress)

    def sense_distance(self, actual_distance: float) -> float:
        """Returns sensed distance with added noise."""
        # add noise proportional to distance (10% standard deviation)
        noise = np.random.normal(0, self.sensor_noise_level * actual_distance * 0.1)
        sensed_distance = actual_distance + noise
        # make sure distance is at least 0.1
        return max(0.1, sensed_distance)

    def update_energy_and_coverage(self, dt: float = 0.1) -> None:
        """Updates energy consumption and current coverage area."""
        # update energy based on state
        if self.is_quarantined:
            self.energy_consumed += self.ENERGY_IDLE * dt
        else:
            self.energy_consumed += self.ENERGY_ACTIVE * dt

        # update coverage area if active
        if not self.is_quarantined:
            coverage_radius = 10.0 * self.task_rate
            self.current_coverage_area = np.pi * coverage_radius ** 2
        else:
            self.current_coverage_area = 0.0

    def update_stress_history(self, stress: float) -> None:
        """Maintains a history of stress levels."""
        # add new stress value
        self.stress_history.append(stress)
        # keep only last 20 values
        if len(self.stress_history) > 20:
            self.stress_history.pop(0)

    def get_stress_persistence(self) -> float:
        """Calculates the fraction of recent samples with high stress."""
        # need at least 5 samples
        if len(self.stress_history) < 5:
            return 0.0

        # check last 10 samples for high stress
        stress_threshold = 50.0
        recent_samples = self.stress_history[-10:]
        stressed_samples = 0

        # count stressed samples
        for stress in recent_samples:
            if stress > stress_threshold:
                stressed_samples += 1

        # return fraction
        return stressed_samples / len(recent_samples)

    def get_state(self) -> AgentState:
        """Returns the current agent state."""
        # create state object with current values
        state = AgentState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            task_rate=self.task_rate,
            stress_level=self.stress_level,
            is_quarantined=self.is_quarantined,
            energy_consumed=self.energy_consumed,
            area_covered=self.current_coverage_area
        )
        return state

    def distance_to(self, other: 'BaseAgent') -> float:
        """Computes Euclidean distance to another agent."""
        # calculate distance using numpy
        return np.linalg.norm(self.position - other.position)

    def __repr__(self) -> str:
        """Returns a string representation of the agent."""
        # format agent info as string
        return (
            f"Agent(id={self.agent_id}, "
            f"pos=({self.position[0]:.1f}, {self.position[1]:.1f}), "
            f"fault={self.fault_type.value}, "
            f"task_rate={self.task_rate:.2f}, "
            f"quarantined={self.is_quarantined})"
        )