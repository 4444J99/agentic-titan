"""
Swarm Intelligence Agent - PSO/ACO optimization patterns.

Implements swarm intelligence algorithms:
- Particle Swarm Optimization (PSO): Personal best + global best
- Ant Colony Optimization (ACO): Pheromone-based routing
- Momentum-based exploration to prevent premature convergence

These agents explore solution spaces collectively, using
indirect coordination through the environment or shared state.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from agents.framework.base_agent import BaseAgent, AgentState

if TYPE_CHECKING:
    from hive.stigmergy import PheromoneField

logger = logging.getLogger("titan.agents.swarm_intelligence")


class SwarmAlgorithm(str, Enum):
    """Types of swarm intelligence algorithms."""

    PSO = "pso"  # Particle Swarm Optimization
    ACO = "aco"  # Ant Colony Optimization
    HYBRID = "hybrid"  # Combination


@dataclass
class Particle:
    """A particle in PSO (virtual agent exploring a landscape)."""

    particle_id: str
    position: list[float]  # Current position in search space
    velocity: list[float]  # Current velocity
    personal_best_position: list[float]  # Best position found by this particle
    personal_best_fitness: float = float("-inf")
    current_fitness: float = float("-inf")

    def update_personal_best(self, fitness: float) -> bool:
        """Update personal best if current position is better.

        Args:
            fitness: Fitness at current position.

        Returns:
            True if personal best was updated.
        """
        self.current_fitness = fitness
        if fitness > self.personal_best_fitness:
            self.personal_best_fitness = fitness
            self.personal_best_position = list(self.position)
            return True
        return False


@dataclass
class SwarmState:
    """State of the swarm."""

    global_best_position: list[float] = field(default_factory=list)
    global_best_fitness: float = float("-inf")
    iteration: int = 0
    convergence_history: list[float] = field(default_factory=list)
    stagnation_count: int = 0


@dataclass
class PSOConfig:
    """Configuration for Particle Swarm Optimization."""

    dimensions: int = 2  # Search space dimensions
    num_particles: int = 30
    inertia_weight: float = 0.7  # Momentum (w)
    cognitive_weight: float = 1.5  # Personal best attraction (c1)
    social_weight: float = 1.5  # Global best attraction (c2)
    velocity_clamp: float = 1.0  # Max velocity magnitude
    position_bounds: tuple[float, float] = (-10.0, 10.0)


@dataclass
class ACOConfig:
    """Configuration for Ant Colony Optimization."""

    num_ants: int = 20
    pheromone_evaporation: float = 0.1
    pheromone_deposit: float = 1.0
    alpha: float = 1.0  # Pheromone importance
    beta: float = 2.0  # Heuristic importance


class SwarmIntelligenceAgent(BaseAgent):
    """
    Agent implementing swarm intelligence patterns.

    Uses PSO, ACO, or hybrid approaches for collective optimization:
    - PSO: Particles explore based on personal and global best
    - ACO: Virtual ants deposit pheromones on good paths
    - Both balance exploration and exploitation

    Capabilities:
    - Optimization via swarm algorithms
    - Pheromone-based coordination (ACO)
    - Personal/global best tracking (PSO)
    - Adaptive exploration
    """

    # Stagnation threshold for algorithm switching
    STAGNATION_THRESHOLD = 20

    def __init__(
        self,
        algorithm: SwarmAlgorithm = SwarmAlgorithm.PSO,
        pso_config: PSOConfig | None = None,
        aco_config: ACOConfig | None = None,
        pheromone_field: Any | None = None,
        fitness_function: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize swarm intelligence agent.

        Args:
            algorithm: Swarm algorithm to use.
            pso_config: PSO configuration.
            aco_config: ACO configuration.
            pheromone_field: Shared pheromone field for ACO.
            fitness_function: Function to evaluate solutions.
            **kwargs: Base agent arguments.
        """
        kwargs.setdefault("name", f"swarm_{algorithm.value}")
        kwargs.setdefault("capabilities", [
            "optimization",
            "swarm_coordination",
            "exploration",
            "exploitation",
        ])
        super().__init__(**kwargs)

        self._algorithm = algorithm
        self._pso_config = pso_config or PSOConfig()
        self._aco_config = aco_config or ACOConfig()
        self._pheromone_field = pheromone_field
        self._fitness_function = fitness_function or self._default_fitness

        self._particles: list[Particle] = []
        self._swarm_state = SwarmState()
        self._graph: dict[str, list[str]] = {}  # For ACO

    @property
    def algorithm(self) -> SwarmAlgorithm:
        """Get current algorithm."""
        return self._algorithm

    @property
    def global_best(self) -> tuple[list[float], float]:
        """Get global best position and fitness."""
        return self._swarm_state.global_best_position, self._swarm_state.global_best_fitness

    async def initialize(self) -> None:
        """Initialize swarm intelligence agent."""
        self._state = AgentState.READY

        if self._algorithm in [SwarmAlgorithm.PSO, SwarmAlgorithm.HYBRID]:
            self._initialize_particles()

        logger.info(f"Swarm intelligence agent initialized (algorithm={self._algorithm.value})")

    def _initialize_particles(self) -> None:
        """Initialize PSO particles."""
        config = self._pso_config
        self._particles = []

        for i in range(config.num_particles):
            # Random initial position within bounds
            position = [
                random.uniform(config.position_bounds[0], config.position_bounds[1])
                for _ in range(config.dimensions)
            ]

            # Random initial velocity
            velocity = [
                random.uniform(-config.velocity_clamp, config.velocity_clamp)
                for _ in range(config.dimensions)
            ]

            particle = Particle(
                particle_id=f"particle_{i}",
                position=position,
                velocity=velocity,
                personal_best_position=list(position),
            )

            # Evaluate initial fitness
            fitness = self._fitness_function(position)
            particle.update_personal_best(fitness)

            self._particles.append(particle)

            # Update global best if needed
            self._update_global_best(particle)

    async def work(self) -> dict[str, Any]:
        """Perform swarm optimization iteration."""
        result = {
            "algorithm": self._algorithm.value,
            "iteration": self._swarm_state.iteration,
            "global_best_fitness": self._swarm_state.global_best_fitness,
        }

        if self._algorithm == SwarmAlgorithm.PSO:
            result.update(await self._pso_iteration())
        elif self._algorithm == SwarmAlgorithm.ACO:
            result.update(await self._aco_iteration())
        else:  # HYBRID
            result.update(await self._hybrid_iteration())

        self._swarm_state.iteration += 1
        self._swarm_state.convergence_history.append(self._swarm_state.global_best_fitness)

        # Check for stagnation
        if len(self._swarm_state.convergence_history) > 2:
            if self._swarm_state.convergence_history[-1] == self._swarm_state.convergence_history[-2]:
                self._swarm_state.stagnation_count += 1
            else:
                self._swarm_state.stagnation_count = 0

        result["stagnation_count"] = self._swarm_state.stagnation_count

        return result

    async def shutdown(self) -> None:
        """Shutdown swarm intelligence agent."""
        logger.info(
            f"Swarm intelligence shutdown "
            f"(best_fitness={self._swarm_state.global_best_fitness:.4f})"
        )

    # =========================================================================
    # Particle Swarm Optimization
    # =========================================================================

    async def _pso_iteration(self) -> dict[str, Any]:
        """Perform one PSO iteration.

        Returns:
            Iteration results.
        """
        config = self._pso_config
        improvements = 0
        avg_fitness = 0.0

        for particle in self._particles:
            # Update velocity
            self._update_velocity(particle)

            # Update position
            self._update_position(particle)

            # Evaluate fitness
            fitness = self._fitness_function(particle.position)
            avg_fitness += fitness

            # Update personal and global best
            if particle.update_personal_best(fitness):
                improvements += 1
                self._update_global_best(particle)

        return {
            "particles_improved": improvements,
            "average_fitness": avg_fitness / len(self._particles),
        }

    def _update_velocity(self, particle: Particle) -> None:
        """Update particle velocity based on PSO equations."""
        config = self._pso_config

        for d in range(config.dimensions):
            r1, r2 = random.random(), random.random()

            # Velocity update: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
            cognitive = config.cognitive_weight * r1 * (
                particle.personal_best_position[d] - particle.position[d]
            )
            social = config.social_weight * r2 * (
                self._swarm_state.global_best_position[d] - particle.position[d]
                if self._swarm_state.global_best_position else 0
            )

            particle.velocity[d] = (
                config.inertia_weight * particle.velocity[d] +
                cognitive +
                social
            )

            # Clamp velocity
            particle.velocity[d] = max(
                -config.velocity_clamp,
                min(config.velocity_clamp, particle.velocity[d])
            )

    def _update_position(self, particle: Particle) -> None:
        """Update particle position."""
        config = self._pso_config

        for d in range(config.dimensions):
            particle.position[d] += particle.velocity[d]

            # Clamp position to bounds
            particle.position[d] = max(
                config.position_bounds[0],
                min(config.position_bounds[1], particle.position[d])
            )

    def _update_global_best(self, particle: Particle) -> bool:
        """Update global best if particle's personal best is better.

        Args:
            particle: Particle to check.

        Returns:
            True if global best was updated.
        """
        if particle.personal_best_fitness > self._swarm_state.global_best_fitness:
            self._swarm_state.global_best_fitness = particle.personal_best_fitness
            self._swarm_state.global_best_position = list(particle.personal_best_position)
            return True
        return False

    # =========================================================================
    # Ant Colony Optimization
    # =========================================================================

    async def _aco_iteration(self) -> dict[str, Any]:
        """Perform one ACO iteration.

        Returns:
            Iteration results.
        """
        config = self._aco_config
        results = {
            "ants_completed": 0,
            "best_path_length": float("inf"),
            "pheromone_deposited": 0,
        }

        if not self._pheromone_field:
            logger.warning("ACO requires pheromone field")
            return results

        # Simulate ants constructing solutions
        for ant in range(config.num_ants):
            path, path_quality = await self._construct_ant_path()

            if path_quality > 0:
                results["ants_completed"] += 1

                # Deposit pheromone proportional to quality
                await self._deposit_pheromone(path, path_quality)
                results["pheromone_deposited"] += 1

                if 1.0 / path_quality < results["best_path_length"]:
                    results["best_path_length"] = 1.0 / path_quality

        # Evaporate pheromones
        await self._pheromone_field.decay_cycle()

        return results

    async def _construct_ant_path(self) -> tuple[list[str], float]:
        """Construct a path using ant behavior.

        Returns:
            Tuple of (path, path_quality).
        """
        if not self._graph:
            return [], 0.0

        config = self._aco_config
        path: list[str] = []
        visited: set[str] = set()

        # Start from a random node
        current = random.choice(list(self._graph.keys()))
        path.append(current)
        visited.add(current)

        # Construct path
        while True:
            neighbors = [n for n in self._graph.get(current, []) if n not in visited]
            if not neighbors:
                break

            # Select next node probabilistically
            next_node = await self._select_next_node(current, neighbors)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        # Calculate path quality (inverse of length for TSP-like problems)
        path_quality = 1.0 / len(path) if path else 0.0

        return path, path_quality

    async def _select_next_node(
        self,
        current: str,
        candidates: list[str],
    ) -> str:
        """Select next node using pheromone and heuristic information.

        Args:
            current: Current node.
            candidates: Candidate next nodes.

        Returns:
            Selected next node.
        """
        config = self._aco_config

        if not self._pheromone_field:
            return random.choice(candidates)

        from hive.stigmergy import TraceType

        # Calculate selection probabilities
        probabilities: list[float] = []

        for candidate in candidates:
            # Get pheromone level
            traces = await self._pheromone_field.sense(
                location=f"{current}_{candidate}",
                trace_types=[TraceType.PATH],
            )
            pheromone = sum(t.intensity for t in traces) + 0.01  # Avoid zero

            # Heuristic (e.g., inverse distance - simplified to 1.0)
            heuristic = 1.0

            # Probability = pheromone^alpha * heuristic^beta
            prob = (pheromone ** config.alpha) * (heuristic ** config.beta)
            probabilities.append(prob)

        # Normalize and select
        total = sum(probabilities)
        if total == 0:
            return random.choice(candidates)

        probabilities = [p / total for p in probabilities]

        # Roulette wheel selection
        r = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return candidates[i]

        return candidates[-1]

    async def _deposit_pheromone(
        self,
        path: list[str],
        quality: float,
    ) -> None:
        """Deposit pheromone along a path.

        Args:
            path: Path to deposit on.
            quality: Quality of the path.
        """
        if not self._pheromone_field:
            return

        from hive.stigmergy import TraceType

        config = self._aco_config
        deposit_amount = config.pheromone_deposit * quality

        for i in range(len(path) - 1):
            edge = f"{path[i]}_{path[i+1]}"
            await self._pheromone_field.deposit(
                agent_id=self.agent_id,
                trace_type=TraceType.PATH,
                location=edge,
                intensity=deposit_amount,
                payload={"quality": quality},
            )

    # =========================================================================
    # Hybrid Algorithm
    # =========================================================================

    async def _hybrid_iteration(self) -> dict[str, Any]:
        """Perform hybrid PSO/ACO iteration.

        Uses PSO for continuous optimization and ACO for
        discrete/combinatorial aspects.

        Returns:
            Iteration results.
        """
        result = {}

        # PSO exploration
        pso_result = await self._pso_iteration()
        result.update({f"pso_{k}": v for k, v in pso_result.items()})

        # Check for stagnation
        if self._swarm_state.stagnation_count > self.STAGNATION_THRESHOLD:
            # Inject randomness to escape local optima
            await self._reset_particles_partial()
            result["reset_particles"] = True
            self._swarm_state.stagnation_count = 0

        # If pheromone field available, use ACO for path refinement
        if self._pheromone_field and self._graph:
            aco_result = await self._aco_iteration()
            result.update({f"aco_{k}": v for k, v in aco_result.items()})

        return result

    async def _reset_particles_partial(self, fraction: float = 0.3) -> None:
        """Reset a fraction of particles to escape local optima.

        Args:
            fraction: Fraction of particles to reset.
        """
        num_reset = int(len(self._particles) * fraction)
        particles_to_reset = random.sample(self._particles, num_reset)

        config = self._pso_config

        for particle in particles_to_reset:
            # Reset to random position
            particle.position = [
                random.uniform(config.position_bounds[0], config.position_bounds[1])
                for _ in range(config.dimensions)
            ]
            particle.velocity = [
                random.uniform(-config.velocity_clamp, config.velocity_clamp)
                for _ in range(config.dimensions)
            ]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _default_fitness(self, position: list[float]) -> float:
        """Default fitness function (sphere function - maximize negative).

        Args:
            position: Position to evaluate.

        Returns:
            Fitness value.
        """
        # Sphere function (minimize sum of squares) -> negate for maximization
        return -sum(x ** 2 for x in position)

    def set_fitness_function(self, func: Any) -> None:
        """Set a custom fitness function.

        Args:
            func: Function taking position list, returning fitness float.
        """
        self._fitness_function = func

    def set_graph(self, graph: dict[str, list[str]]) -> None:
        """Set the graph for ACO.

        Args:
            graph: Dict mapping node -> list of neighbor nodes.
        """
        self._graph = graph

    def get_convergence_history(self) -> list[float]:
        """Get history of global best fitness values.

        Returns:
            List of fitness values over iterations.
        """
        return list(self._swarm_state.convergence_history)

    def get_statistics(self) -> dict[str, Any]:
        """Get swarm statistics.

        Returns:
            Dictionary with swarm statistics.
        """
        return {
            "algorithm": self._algorithm.value,
            "iterations": self._swarm_state.iteration,
            "global_best_fitness": self._swarm_state.global_best_fitness,
            "global_best_position": self._swarm_state.global_best_position,
            "num_particles": len(self._particles),
            "stagnation_count": self._swarm_state.stagnation_count,
            "convergence_history_length": len(self._swarm_state.convergence_history),
        }
