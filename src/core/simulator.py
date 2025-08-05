"""
Base simulator for swarm fault tolerance experiments.

This module provides an abstract framework for
simulating multi-agent swarms, with extensible
hooks for implementing specific fault detection
strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import asyncio
import time
import random
from dataclasses import dataclass
from enum import Enum

from agent import BaseAgent, FaultType


class ScalingMode(Enum):
    """Defines arena scaling modes."""
    FIXED_ARENA = "fixed_arena"
    FIXED_DENSITY = "fixed_density"


@dataclass
class SimulationConfig:
    """Configuration parameters for a simulation run."""
    num_agents: int
    fault_rate: float
    run_time: float = 30.0
    scaling_mode: ScalingMode = ScalingMode.FIXED_ARENA
    enable_quarantine: bool = True
    packet_loss_rate: float = 0.0
    sensor_noise_level: float = 0.0
    random_seed: Optional[int] = None


@dataclass
class SimulationMetrics:
    """Stores metrics from a simulation run."""
    num_agents: int
    fault_rate: float
    arena_size: float
    density: float
    scaling_mode: str
    task_completion: float
    avg_healthy_performance: float
    mission_success: bool
    severely_impaired: int
    num_healthy: int
    num_faulty: int
    quarantined: int
    quarantine_efficiency: float
    cascade_prevention: float
    avg_response_time: float
    work_per_energy: float
    coverage_efficiency: float
    healthy_preservation_rate: float
    active_agent_efficiency: float
    packet_loss_rate: float
    sensor_noise_level: float
    quarantine_enabled: bool


class BaseSimulator(ABC):
    """
    Abstract base class for swarm simulations.

    Provides core simulation logic for multi-agent systems, including agent initialization, state updates, and metrics computation, with abstract methods for fault detection strategies.

    Attributes:
        config: Simulation configuration.
        BASE_ARENA_SIZE: Default arena size for fixed arena mode (m).
        BASE_DENSITY: Default density for fixed density mode (agents/m²).
        COMMUNICATION_RANGE: Range for agent communication (m).
        IMPEDANCE_RANGE: Range for fault impedance effects (m).
    """

    BASE_ARENA_SIZE = 100.0
    BASE_DENSITY = 0.005
    COMMUNICATION_RANGE = 25.0
    IMPEDANCE_RANGE = 20.0

    def __init__(self, config: SimulationConfig):
        """Initializes the simulator with the given configuration."""
        self.config = config

        # set random seeds if provided
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)

        self.agents: List[BaseAgent] = []

    def calculate_arena_size(self) -> float:
        """Computes arena size based on scaling mode."""
        # check which mode we're using
        if self.config.scaling_mode == ScalingMode.FIXED_ARENA:
            # use fixed size
            return self.BASE_ARENA_SIZE
        else:
            # calculate based on density
            area_needed = self.config.num_agents / self.BASE_DENSITY
            arena_size = np.sqrt(area_needed)
            return arena_size

    def calculate_density(self, arena_size: float) -> float:
        """Computes swarm density (agents/m²)."""
        # calculate area
        area = arena_size * arena_size
        # calculate density
        density = self.config.num_agents / area
        return density

    def create_agents(self) -> List[BaseAgent]:
        """Initializes agents with distributed faults and grid-based positions."""
        arena_size = self.calculate_arena_size()

        # determine how many faulty agents
        num_faulty = int(self.config.num_agents * self.config.fault_rate)

        # assign fault types
        fault_assignments = self._assign_fault_types(num_faulty)

        # create grid layout
        grid_size = int(np.ceil(np.sqrt(self.config.num_agents)))
        if grid_size > 1:
            spacing = arena_size / (grid_size - 1)
        else:
            spacing = arena_size

        # randomly select which agents will be faulty
        fault_indices = set(random.sample(range(self.config.num_agents), num_faulty))

        # create agents
        agents = []
        fault_counter = 0

        for i in range(self.config.num_agents):
            # calculate grid position
            row = i // grid_size
            col = i % grid_size

            # add some randomness to position
            x_pos = col * spacing + random.uniform(-5, 5)
            y_pos = row * spacing + random.uniform(-5, 5)

            # keep within bounds
            x_pos = np.clip(x_pos, 10, arena_size - 10)
            y_pos = np.clip(y_pos, 10, arena_size - 10)
            position = (x_pos, y_pos)

            # determine fault type
            if i in fault_indices:
                fault_type = fault_assignments[fault_counter]
                fault_counter += 1
            else:
                fault_type = FaultType.NONE

            # create agent
            agent = self._create_agent(i, position, fault_type, arena_size)
            agents.append(agent)

        self.agents = agents
        return agents

    def _assign_fault_types(self, num_faulty: int) -> List[FaultType]:
        """Assigns fault types with a realistic distribution."""
        # handle no faults case
        if num_faulty == 0:
            return []

        # distribute fault types
        num_severe = max(1, int(num_faulty * 0.15))
        num_moderate = int(num_faulty * 0.35)
        num_minor = num_faulty - num_severe - num_moderate

        # create list of fault types
        fault_assignments = []

        # add severe faults
        for i in range(num_severe):
            fault_assignments.append(FaultType.SEVERE)

        # add moderate faults
        for i in range(num_moderate):
            fault_assignments.append(FaultType.MODERATE)

        # add minor faults
        for i in range(num_minor):
            fault_assignments.append(FaultType.MINOR)

        # shuffle the list
        random.shuffle(fault_assignments)
        return fault_assignments

    def _create_agent(self, agent_id: int, position: Tuple[float, float], fault_type: FaultType, arena_size: float) -> BaseAgent:
        """Creates a single agent with specified parameters."""
        # create agent with all parameters
        agent = BaseAgent(
            agent_id=agent_id,
            position=position,
            fault_type=fault_type,
            arena_size=arena_size,
            packet_loss_rate=self.config.packet_loss_rate,
            sensor_noise_level=self.config.sensor_noise_level
        )
        return agent

    async def run_simulation(self) -> SimulationMetrics:
        """Executes the simulation and returns final metrics."""
        # calculate arena parameters
        arena_size = self.calculate_arena_size()
        density = self.calculate_density(arena_size)

        # create agents
        self.create_agents()

        # print configuration
        self._print_configuration(arena_size, density)

        # initialize fault detection
        self.initialize_fault_detection(density)

        # start timing
        start_time = time.time()
        steps = int(self.config.run_time * 10)  # 10 steps per second

        # run simulation loop
        for step in range(steps):
            elapsed = time.time() - start_time

            # update agent states
            self._update_agent_states()

            # update interactions between agents
            self._update_agent_interactions(density)

            # update fault detection if enabled
            if self.config.enable_quarantine:
                await self.update_fault_detection(elapsed, density)

            # print progress occasionally
            if step % 50 == 0 or step == steps - 1:
                self._print_progress(step, elapsed)

            # small delay
            await asyncio.sleep(0.001)

        # calculate final metrics
        metrics = self._calculate_metrics(arena_size, density)

        # print results
        self._print_results(metrics)

        return metrics

    def _update_agent_states(self) -> None:
        """Updates agent positions and energy metrics."""
        # update each agent
        for agent in self.agents:
            # only move if not quarantined
            if not agent.is_quarantined:
                agent.update_position()

            # always update energy and coverage
            agent.update_energy_and_coverage()

    def _update_agent_interactions(self, density: float) -> None:
        """Updates agent interactions, including stress and damage propagation."""
        # get adaptive parameters
        params = self.get_adaptive_parameters(density)

        # reset agent states
        for agent in self.agents:
            agent.faulty_neighbors = 0
            agent.stress_level = 0

        # check all pairs of agents
        for i in range(len(self.agents)):
            agent1 = self.agents[i]

            # skip if quarantined
            if agent1.is_quarantined:
                continue

            # check against all other agents
            for j in range(i + 1, len(self.agents)):
                agent2 = self.agents[j]

                # skip if quarantined
                if agent2.is_quarantined:
                    continue

                # calculate distance
                actual_dist = agent1.distance_to(agent2)

                # agent1 senses distance to agent2
                sensed_dist1 = agent1.sense_distance(actual_dist)
                # agent2 senses distance to agent1
                sensed_dist2 = agent2.sense_distance(actual_dist)

                # check if agent2 affects agent1 (impedance)
                if sensed_dist1 <= self.IMPEDANCE_RANGE and agent2.is_faulty and not agent1.is_faulty:
                    agent1.faulty_neighbors += 1

                    # calculate damage
                    proximity_factor = np.exp(-sensed_dist1 / 10)
                    crowding_factor = 1 + min(agent1.faulty_neighbors * 0.15, 0.5)

                    # severity based on fault type
                    if agent2.fault_type == FaultType.SEVERE:
                        fault_severity = 0.8
                    else:
                        fault_severity = 0.5

                    damage = params['damage_rate'] * proximity_factor * crowding_factor * fault_severity * 0.7
                    agent1.take_damage(damage)

                    # update time tracking
                    agent1.time_near_faulty += 0.1
                    agent1.time_away_from_faulty = 0

                # check if agent1 affects agent2 (impedance)
                if sensed_dist2 <= self.IMPEDANCE_RANGE and agent1.is_faulty and not agent2.is_faulty:
                    agent2.faulty_neighbors += 1

                    # calculate damage
                    proximity_factor = np.exp(-sensed_dist2 / 10)
                    crowding_factor = 1 + min(agent2.faulty_neighbors * 0.15, 0.5)

                    # severity based on fault type
                    if agent1.fault_type == FaultType.SEVERE:
                        fault_severity = 0.8
                    else:
                        fault_severity = 0.5

                    damage = params['damage_rate'] * proximity_factor * crowding_factor * fault_severity * 0.7
                    agent2.take_damage(damage)

                    # update time tracking
                    agent2.time_near_faulty += 0.1
                    agent2.time_away_from_faulty = 0

                # check if agent2 stress affects agent1
                if sensed_dist1 <= self.COMMUNICATION_RANGE and agent2.is_faulty:
                    sensed_stress = agent1.sense_neighbor_stress(agent2.stress_output)
                    if sensed_stress > 0:
                        stress_contrib = sensed_stress * np.exp(-sensed_dist1 / 18)

                        # adjust for density in fixed arena mode
                        if self.config.scaling_mode == ScalingMode.FIXED_ARENA and density > 0.005:
                            density_factor = min(1 + (density - 0.005) * 15, 1.5)
                            stress_contrib *= density_factor

                        agent1.stress_level += stress_contrib

                # check if agent1 stress affects agent2
                if sensed_dist2 <= self.COMMUNICATION_RANGE and agent1.is_faulty:
                    sensed_stress = agent2.sense_neighbor_stress(agent1.stress_output)
                    if sensed_stress > 0:
                        stress_contrib = sensed_stress * np.exp(-sensed_dist2 / 18)

                        # adjust for density in fixed arena mode
                        if self.config.scaling_mode == ScalingMode.FIXED_ARENA and density > 0.005:
                            density_factor = min(1 + (density - 0.005) * 15, 1.5)
                            stress_contrib *= density_factor

                        agent2.stress_level += stress_contrib

        # handle recovery for agents away from faulty ones
        for agent in self.agents:
            if agent.faulty_neighbors == 0:
                agent.time_away_from_faulty += 0.1

                # apply recovery if away long enough
                if agent.time_away_from_faulty > 1.5:
                    agent.apply_recovery(params['recovery_rate'])
            else:
                agent.time_away_from_faulty = 0

            # update stress history
            agent.update_stress_history(agent.stress_level)

    def _calculate_metrics(self, arena_size: float, density: float) -> SimulationMetrics:
        """Computes comprehensive simulation metrics."""
        # separate agents by type
        healthy_agents = []
        faulty_agents = []
        active_agents = []

        for agent in self.agents:
            if not agent.is_faulty:
                healthy_agents.append(agent)
            else:
                faulty_agents.append(agent)

            if not agent.is_quarantined:
                active_agents.append(agent)

        # calculate task completion
        if self.config.num_agents > 0:
            total_task_rate = 0.0
            for agent in active_agents:
                total_task_rate += agent.task_rate
            task_completion = total_task_rate / self.config.num_agents
        else:
            task_completion = 0.0

        # calculate average healthy performance
        if healthy_agents:
            healthy_task_sum = 0.0
            healthy_active_count = 0
            for agent in healthy_agents:
                if not agent.is_quarantined:
                    healthy_task_sum += agent.task_rate
                    healthy_active_count += 1

            if healthy_active_count > 0:
                avg_healthy_performance = healthy_task_sum / healthy_active_count
            else:
                avg_healthy_performance = 0.0
        else:
            avg_healthy_performance = 0.0

        # evaluate mission success
        mission_success = self._evaluate_mission_success(healthy_agents, density, self.config.fault_rate)

        # count severely impaired healthy agents
        severely_impaired = 0
        for agent in healthy_agents:
            if agent.task_rate < 0.5:
                severely_impaired += 1

        # count quarantined agents
        quarantined_count = 0
        for agent in self.agents:
            if agent.is_quarantined:
                quarantined_count += 1

        # calculate quarantine efficiency
        if faulty_agents:
            quarantined_faulty = 0
            for agent in faulty_agents:
                if agent.is_quarantined:
                    quarantined_faulty += 1
            quarantine_efficiency = quarantined_faulty / len(faulty_agents)
        else:
            quarantine_efficiency = 0.0

        # calculate cascade prevention
        if healthy_agents:
            cascade_prevention = 1.0 - (severely_impaired / len(healthy_agents))
        else:
            cascade_prevention = 0.0

        # calculate energy metrics
        total_energy = 0.0
        for agent in self.agents:
            total_energy += agent.energy_consumed

        if total_energy > 0:
            total_work = 0.0
            for agent in active_agents:
                total_work += agent.task_rate
            work_per_energy = total_work / total_energy
        else:
            work_per_energy = 0.0

        # calculate coverage
        total_coverage = 0.0
        for agent in active_agents:
            total_coverage += agent.current_coverage_area

        if active_agents:
            coverage_efficiency = total_coverage / len(active_agents)
        else:
            coverage_efficiency = 0.0

        # calculate healthy preservation
        if healthy_agents:
            high_performing = 0
            for agent in healthy_agents:
                if agent.task_rate > 0.9:
                    high_performing += 1
            healthy_preservation_rate = high_performing / len(healthy_agents)
        else:
            healthy_preservation_rate = 0.0

        # calculate active agent efficiency
        if active_agents:
            active_task_sum = 0.0
            for agent in active_agents:
                active_task_sum += agent.task_rate
            active_agent_efficiency = active_task_sum / len(active_agents)
        else:
            active_agent_efficiency = 0.0

        # create metrics object
        metrics = SimulationMetrics(
            num_agents=self.config.num_agents,
            fault_rate=self.config.fault_rate,
            arena_size=arena_size,
            density=density,
            scaling_mode=self.config.scaling_mode.value,
            task_completion=task_completion,
            avg_healthy_performance=avg_healthy_performance,
            mission_success=mission_success,
            severely_impaired=severely_impaired,
            num_healthy=len(healthy_agents),
            num_faulty=len(faulty_agents),
            quarantined=quarantined_count,
            quarantine_efficiency=quarantine_efficiency,
            cascade_prevention=cascade_prevention,
            avg_response_time=self._calculate_response_time(),
            work_per_energy=work_per_energy,
            coverage_efficiency=coverage_efficiency,
            healthy_preservation_rate=healthy_preservation_rate,
            active_agent_efficiency=active_agent_efficiency,
            packet_loss_rate=self.config.packet_loss_rate,
            sensor_noise_level=self.config.sensor_noise_level,
            quarantine_enabled=self.config.enable_quarantine
        )

        return metrics

    def _evaluate_mission_success(self, healthy_agents: List[BaseAgent], density: float, fault_rate: float) -> bool:
        """Evaluates mission success based on healthy agent performance."""
        # no healthy agents means failure
        if not healthy_agents:
            return False

        # determine thresholds based on conditions
        if density > 0.01:
            threshold = 0.6
            required_fraction = 0.7
        elif fault_rate <= 0.10:
            threshold = 0.75
            required_fraction = 0.8
        else:
            threshold = 0.65
            required_fraction = 0.75

        # count agents above threshold
        above_threshold = 0
        for agent in healthy_agents:
            if agent.task_rate > threshold and not agent.is_quarantined:
                above_threshold += 1

        # check if enough agents meet criteria
        success_fraction = above_threshold / len(healthy_agents)
        return success_fraction >= required_fraction

    def _print_configuration(self, arena_size: float, density: float) -> None:
        """Prints simulation configuration details."""
        # count fault types
        fault_counts = {}
        for fault_type in FaultType:
            if fault_type != FaultType.NONE:
                count = 0
                for agent in self.agents:
                    if agent.fault_type == fault_type:
                        count += 1
                if count > 0:
                    fault_counts[fault_type] = count

        # build fault string
        fault_parts = []
        for fault_type, count in fault_counts.items():
            fault_parts.append(f"{count} {fault_type.value}")

        if fault_parts:
            fault_str = ", ".join(fault_parts)
        else:
            fault_str = "None"

        # format scaling mode
        scaling_str = self.config.scaling_mode.value.replace('_', ' ').title()

        # print configuration
        print(f"\nSimulation Configuration:")
        print(f"  Agents: {self.config.num_agents}, Fault Rate: {self.config.fault_rate * 100:.0f}%")
        print(f"  Scaling Mode: {scaling_str}")
        print(f"  Arena Size: {arena_size:.0f}x{arena_size:.0f} m, Density: {density:.4f} agents/m²")
        print(f"  Quarantine: {'Enabled' if self.config.enable_quarantine else 'Disabled'}")
        print(f"  Faults: {fault_str}")
        print(f"  Robustness: Packet Loss = {self.config.packet_loss_rate * 100:.0f}%, Sensor Noise = {self.config.sensor_noise_level * 100:.0f}%")

    def _print_progress(self, step: int, elapsed: float) -> None:
        """Prints simulation progress updates."""
        # count healthy active agents
        healthy_active = []
        for agent in self.agents:
            if not agent.is_faulty and not agent.is_quarantined:
                healthy_active.append(agent)

        # calculate average health
        if healthy_active:
            health_sum = 0.0
            for agent in healthy_active:
                health_sum += agent.task_rate
            avg_health = health_sum / len(healthy_active)
        else:
            avg_health = 0.0

        # count various states
        quarantined = 0
        stressed = 0
        healthy_high_performing = 0
        total_healthy = 0

        for agent in self.agents:
            if agent.is_quarantined:
                quarantined += 1
            if agent.stress_level > 60:
                stressed += 1
            if not agent.is_faulty:
                total_healthy += 1
                if agent.task_rate > 0.9:
                    healthy_high_performing += 1

        # print progress
        print(f"Progress t={step / 10:.0f}s: "
              f"Quarantined={quarantined}, "
              f"Stressed={stressed}, "
              f"Healthy={healthy_high_performing}/{total_healthy}, "
              f"Avg Health={avg_health:.2f}")

    def _print_results(self, metrics: SimulationMetrics) -> None:
        """Prints final simulation results."""
        print(f"\nSimulation Results:")
        print(f"  Task Completion: {metrics.task_completion * 100:.1f}%")
        print(f"  Avg Healthy Performance: {metrics.avg_healthy_performance * 100:.1f}%")
        print(f"  Severely Impaired: {metrics.severely_impaired}/{metrics.num_healthy}")
        print(f"  Quarantined: {metrics.quarantined} ({metrics.quarantine_efficiency * 100:.1f}% of faulty)")
        print(f"  Mission Success: {'Yes' if metrics.mission_success else 'No'}")
        print(f"  Cascade Prevention: {metrics.cascade_prevention * 100:.1f}%")
        print(f"  Efficiency Metrics:")
        print(f"    Work per Energy: {metrics.work_per_energy:.2f}")
        print(f"    Coverage Efficiency: {metrics.coverage_efficiency:.1f} m²/agent")
        print(f"    Healthy Preservation: {metrics.healthy_preservation_rate * 100:.0f}%")
        print(f"    Active Efficiency: {metrics.active_agent_efficiency:.2f}")

    @abstractmethod
    def get_adaptive_parameters(self, density: float) -> Dict[str, Any]:
        """Returns parameters adjusted for swarm density."""
        pass

    @abstractmethod
    def initialize_fault_detection(self, density: float) -> None:
        """Initializes fault detection strategy."""
        pass

    @abstractmethod
    async def update_fault_detection(self, elapsed: float, density: float) -> None:
        """Updates fault detection and quarantine decisions."""
        pass

    @abstractmethod
    def _calculate_response_time(self) -> float:
        """Calculates average fault detection response time."""
        pass