"""
Evaluation metrics for swarm fault tolerance experiments.

This module provides metrics to assess the performance
of fault detection strategies in agent-based swarm
simulations, focusing on task completion, quarantine
efficiency, energy efficiency, and fault cascade
prevention.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from agent import BaseAgent, FaultType


@dataclass
class MetricHistory:
    """Stores metric values over time for analysis."""
    timestamps: List[float] = field(default_factory=list)
    tcr_values: List[float] = field(default_factory=list)
    qe_values: List[float] = field(default_factory=list)
    wpe_values: List[float] = field(default_factory=list)
    cpr_values: List[float] = field(default_factory=list)

    def add_sample(self, timestamp: float, tcr: float, qe: float, wpe: float, cpr: float) -> None:
        """Records a metric sample at the specified timestamp."""
        # add all values to their lists
        self.timestamps.append(timestamp)
        self.tcr_values.append(tcr)
        self.qe_values.append(qe)
        self.wpe_values.append(wpe)
        self.cpr_values.append(cpr)


class SwarmMetrics:
    """
    Computes primary evaluation metrics for swarm performance.

    Metrics include:
    - Task Completion Rate (TCR): Average task efficiency.
    - Quarantine Efficiency (QE): Fraction of faulty agents quarantined.
    - Work Per Energy (WPE): Work output per unit energy.
    - Cascade Prevention Rate (CPR): Fraction of healthy agents unimpaired.
    """

    # Energy constants - must match agent.py
    ENERGY_IDLE = 0.1  # Power consumption for idle/quarantined agents (W)
    ENERGY_ACTIVE = 1.2  # Power consumption for active agents (W)
    IMPAIRMENT_THRESHOLD = 0.5  # Task rate threshold for impairment

    def __init__(self):
        """Initializes the metrics calculator."""
        self.history = MetricHistory()

    def calculate_task_completion_rate(self, agents: List[BaseAgent]) -> float:
        """Computes the average task completion rate across the swarm."""
        # handle empty list
        if not agents:
            return 0.0

        # sum up all task rates
        total_task_rate = 0.0
        for agent in agents:
            total_task_rate += agent.task_rate

        # calculate average
        average_rate = total_task_rate / len(agents)
        return average_rate

    def calculate_quarantine_efficiency(self, agents: List[BaseAgent]) -> float:
        """Computes the fraction of faulty agents successfully quarantined."""
        # find all faulty agents
        faulty_agents = []
        for agent in agents:
            if agent.is_faulty:
                faulty_agents.append(agent)

        # if no faulty agents, perfect efficiency
        if len(faulty_agents) == 0:
            return 1.0

        # count quarantined faulty agents
        quarantined_count = 0
        for agent in faulty_agents:
            if agent.is_quarantined:
                quarantined_count += 1

        # calculate efficiency
        efficiency = quarantined_count / len(faulty_agents)
        return efficiency

    def calculate_work_per_energy(self, agents: List[BaseAgent], dt: float = 0.1) -> float:
        """Computes instantaneous work per energy (work units per Joule)."""
        # handle empty list
        if not agents:
            return 0.0

        # calculate total work done
        total_work = 0.0
        for agent in agents:
            if not agent.is_quarantined:
                total_work += agent.task_rate * dt

        # calculate total energy consumed
        total_energy = 0.0
        for agent in agents:
            if agent.is_quarantined:
                total_energy += self.ENERGY_IDLE * dt
            else:
                total_energy += self.ENERGY_ACTIVE * dt

        # avoid division by zero
        if total_energy > 0:
            return total_work / total_energy
        else:
            return 0.0

    def calculate_cumulative_work_per_energy(self, agents: List[BaseAgent]) -> float:
        """Computes cumulative work per energy using stored energy values."""
        # handle empty list
        if not agents:
            return 0.0

        # calculate total energy consumed
        total_energy = 0.0
        for agent in agents:
            total_energy += agent.energy_consumed

        # avoid division by zero
        if total_energy == 0:
            return 0.0

        # calculate total work done
        total_work = 0.0
        for agent in agents:
            # estimate time spent based on energy and power consumption
            if agent.is_quarantined:
                time_spent = agent.energy_consumed / self.ENERGY_IDLE
            else:
                time_spent = agent.energy_consumed / self.ENERGY_ACTIVE

            # work = task_rate * time
            work_done = agent.task_rate * time_spent
            total_work += work_done

        # return work per energy
        return total_work / total_energy

    def calculate_cascade_prevention_rate(self, agents: List[BaseAgent]) -> float:
        """Computes the fraction of healthy agents with task rate above impairment threshold."""
        # find all healthy agents
        healthy_agents = []
        for agent in agents:
            if not agent.is_faulty:
                healthy_agents.append(agent)

        # if no healthy agents
        if len(healthy_agents) == 0:
            return 0.0

        # count impaired healthy agents
        impaired_count = 0
        for agent in healthy_agents:
            if agent.task_rate < self.IMPAIRMENT_THRESHOLD:
                impaired_count += 1

        # calculate prevention rate
        prevention_rate = 1.0 - (impaired_count / len(healthy_agents))
        return prevention_rate

    def calculate_all_metrics(self, agents: List[BaseAgent], timestamp: Optional[float] = None, store_history: bool = True) -> Dict[str, float]:
        """Computes all primary metrics and optionally stores them."""
        # calculate each metric
        tcr = self.calculate_task_completion_rate(agents)
        qe = self.calculate_quarantine_efficiency(agents)
        wpe = self.calculate_work_per_energy(agents)
        cpr = self.calculate_cascade_prevention_rate(agents)

        # store in history if requested
        if store_history and timestamp is not None:
            self.history.add_sample(timestamp, tcr, qe, wpe, cpr)

        # return as dictionary
        metrics_dict = {
            'task_completion_rate': tcr,
            'quarantine_efficiency': qe,
            'work_per_energy': wpe,
            'cascade_prevention_rate': cpr
        }
        return metrics_dict

    def get_additional_metrics(self, agents: List[BaseAgent]) -> Dict[str, Any]:
        """Computes additional metrics for detailed swarm analysis."""
        # separate agents by type
        healthy_agents = []
        faulty_agents = []
        active_agents = []

        for agent in agents:
            if not agent.is_faulty:
                healthy_agents.append(agent)
            else:
                faulty_agents.append(agent)

            if not agent.is_quarantined:
                active_agents.append(agent)

        # collect task rates and stress levels
        task_rates = []
        stress_levels = []
        for agent in agents:
            task_rates.append(agent.task_rate)
            stress_levels.append(agent.stress_level)

        # calculate total coverage
        total_coverage = 0.0
        for agent in active_agents:
            total_coverage += agent.current_coverage_area

        # count quarantined agents
        num_quarantined = 0
        for agent in agents:
            if agent.is_quarantined:
                num_quarantined += 1

        # calculate false positive rate
        false_positives = 0
        for agent in healthy_agents:
            if agent.is_quarantined:
                false_positives += 1

        if len(healthy_agents) > 0:
            false_positive_rate = false_positives / len(healthy_agents)
        else:
            false_positive_rate = 0.0

        # calculate false negative rate
        false_negatives = 0
        for agent in faulty_agents:
            if not agent.is_quarantined:
                false_negatives += 1

        if len(faulty_agents) > 0:
            false_negative_rate = false_negatives / len(faulty_agents)
        else:
            false_negative_rate = 0.0

        # count highly stressed agents
        highly_stressed = 0
        for stress in stress_levels:
            if stress > 50:
                highly_stressed += 1

        # calculate averages
        if len(task_rates) > 0:
            avg_task_rate = np.mean(task_rates)
            task_rate_std = np.std(task_rates)
        else:
            avg_task_rate = 0.0
            task_rate_std = 0.0

        if len(healthy_agents) > 0:
            healthy_task_rates = []
            for agent in healthy_agents:
                healthy_task_rates.append(agent.task_rate)
            avg_healthy_task_rate = np.mean(healthy_task_rates)
        else:
            avg_healthy_task_rate = 0.0

        if len(stress_levels) > 0:
            avg_stress_level = np.mean(stress_levels)
        else:
            avg_stress_level = 0.0

        if len(active_agents) > 0:
            coverage_per_agent = total_coverage / len(active_agents)
        else:
            coverage_per_agent = 0.0

        # create results dictionary
        results = {
            'num_healthy': len(healthy_agents),
            'num_faulty': len(faulty_agents),
            'num_quarantined': num_quarantined,
            'num_active': len(active_agents),
            'avg_task_rate': avg_task_rate,
            'avg_healthy_task_rate': avg_healthy_task_rate,
            'task_rate_std': task_rate_std,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'avg_stress_level': avg_stress_level,
            'highly_stressed_count': highly_stressed,
            'total_coverage': total_coverage,
            'coverage_per_agent': coverage_per_agent
        }

        return results

    def get_fault_type_metrics(self, agents: List[BaseAgent]) -> Dict[str, Dict[str, float]]:
        """Computes metrics segmented by fault type."""
        fault_type_metrics = {}

        # check each fault type
        for fault_type in FaultType:
            # find agents with this fault type
            type_agents = []
            for agent in agents:
                if agent.fault_type == fault_type:
                    type_agents.append(agent)

            # if we have agents of this type
            if len(type_agents) > 0:
                # count quarantined
                quarantined_count = 0
                for agent in type_agents:
                    if agent.is_quarantined:
                        quarantined_count += 1

                # calculate averages
                stress_sum = 0.0
                task_rate_sum = 0.0
                for agent in type_agents:
                    stress_sum += agent.stress_level
                    task_rate_sum += agent.task_rate

                # create metrics for this fault type
                type_metrics = {
                    'count': len(type_agents),
                    'quarantined': quarantined_count,
                    'quarantine_rate': quarantined_count / len(type_agents),
                    'avg_stress': stress_sum / len(type_agents),
                    'avg_task_rate': task_rate_sum / len(type_agents)
                }

                # add to results
                fault_type_metrics[fault_type.value] = type_metrics

        return fault_type_metrics

    def calculate_response_metrics(self, detection_times: Dict[int, float], quarantine_times: Dict[int, float]) -> Dict[str, float]:
        """Computes response time statistics for fault detection and quarantine."""
        # handle empty detection times
        if not detection_times:
            empty_result = {
                'avg_detection_time': 0.0,
                'avg_quarantine_time': 0.0,
                'avg_response_time': 0.0,
                'min_response_time': 0.0,
                'max_response_time': 0.0
            }
            return empty_result

        # calculate response times
        response_times = []
        for agent_id, detection_time in detection_times.items():
            if agent_id in quarantine_times:
                response_time = quarantine_times[agent_id] - detection_time
                response_times.append(response_time)

        # calculate averages
        detection_time_list = list(detection_times.values())
        avg_detection = np.mean(detection_time_list)

        if quarantine_times:
            quarantine_time_list = list(quarantine_times.values())
            avg_quarantine = np.mean(quarantine_time_list)
        else:
            avg_quarantine = 0.0

        if response_times:
            avg_response = np.mean(response_times)
            min_response = min(response_times)
            max_response = max(response_times)
        else:
            avg_response = 0.0
            min_response = 0.0
            max_response = 0.0

        # create result dictionary
        result = {
            'avg_detection_time': avg_detection,
            'avg_quarantine_time': avg_quarantine,
            'avg_response_time': avg_response,
            'min_response_time': min_response,
            'max_response_time': max_response
        }

        return result

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Computes summary statistics from metric history."""
        # check if we have history
        if not self.history.timestamps:
            return {}

        # create metrics dictionary
        metrics = {
            'task_completion_rate': self.history.tcr_values,
            'quarantine_efficiency': self.history.qe_values,
            'work_per_energy': self.history.wpe_values,
            'cascade_prevention_rate': self.history.cpr_values
        }

        # calculate statistics for each metric
        summary = {}
        for name, values in metrics.items():
            if values:  # make sure we have values
                # calculate stats
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = min(values)
                max_val = max(values)
                final_val = values[-1]

                # store stats
                summary[name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'final': final_val
                }

        return summary