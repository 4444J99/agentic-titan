"""
Orchestrator Agent - Coordinates multi-agent workflows.

Capabilities:
- Task decomposition
- Agent selection
- Workflow management
- Result aggregation
- Topology-aware task routing
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from agents.framework.base_agent import BaseAgent, AgentResult, AgentState
from agents.personas import ORCHESTRATOR, say, think, announce
from adapters.base import LLMMessage
from adapters.router import get_router

if TYPE_CHECKING:
    from hive.topology import TopologyEngine, BaseTopology

logger = logging.getLogger("titan.agents.orchestrator")


class ExecutionMode(Enum):
    """Execution modes for subtasks."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    STAGED = "staged"  # Pipeline topology: stage by stage
    BROADCAST = "broadcast"  # Swarm topology: all agents work together


@dataclass
class Subtask:
    """A subtask in the workflow."""

    id: str
    description: str
    agent_type: str
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    result: Any = None
    priority: int = 0  # Higher priority executed first (within same dependency level)
    estimated_tokens: int = 0  # Estimated token usage
    assigned_agent_id: str | None = None  # Actual agent assigned


@dataclass
class DependencyGraph:
    """Dependency graph for subtask execution ordering."""

    subtasks: dict[str, Subtask]
    adjacency: dict[str, list[str]]  # subtask_id -> dependent subtask ids

    @classmethod
    def from_subtasks(cls, subtasks: list[Subtask]) -> "DependencyGraph":
        """Build dependency graph from subtask list."""
        task_map = {st.id: st for st in subtasks}
        adjacency: dict[str, list[str]] = {st.id: [] for st in subtasks}

        for st in subtasks:
            for dep_id in st.dependencies:
                if dep_id in adjacency:
                    adjacency[dep_id].append(st.id)

        return cls(subtasks=task_map, adjacency=adjacency)

    def get_ready_tasks(self, completed: set[str]) -> list[Subtask]:
        """Get tasks whose dependencies are all completed."""
        ready = []
        for st_id, subtask in self.subtasks.items():
            if subtask.status != "pending":
                continue
            if all(dep_id in completed for dep_id in subtask.dependencies):
                ready.append(subtask)
        # Sort by priority (higher first)
        return sorted(ready, key=lambda x: x.priority, reverse=True)

    def topological_sort(self) -> list[list[str]]:
        """Return tasks grouped by execution level (for staged execution)."""
        in_degree = {st_id: len(st.dependencies) for st_id, st in self.subtasks.items()}
        levels: list[list[str]] = []
        remaining = set(in_degree.keys())

        while remaining:
            # Find all tasks with no remaining dependencies
            level = [st_id for st_id in remaining if in_degree[st_id] == 0]
            if not level:
                # Cycle detected or all done
                break
            levels.append(level)
            for st_id in level:
                remaining.remove(st_id)
                for dependent in self.adjacency.get(st_id, []):
                    if dependent in in_degree:
                        in_degree[dependent] -= 1

        return levels


@dataclass
class Workflow:
    """A multi-agent workflow."""

    task: str
    subtasks: list[Subtask] = field(default_factory=list)
    topology: str = "pipeline"
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    status: str = "pending"
    results: dict[str, Any] = field(default_factory=dict)
    execution_levels: list[list[str]] = field(default_factory=list)  # For staged execution


class OrchestratorAgent(BaseAgent):
    """
    Agent that coordinates multi-agent workflows.

    Responsibilities:
    - Decompose tasks into subtasks
    - Select appropriate agents
    - Manage topology
    - Monitor progress
    - Aggregate results
    """

    def __init__(
        self,
        task: str | None = None,
        available_agents: list[str] | None = None,
        topology_engine: Any | None = None,
        **kwargs: Any,
    ) -> None:
        # Set defaults that can be overridden by kwargs
        kwargs.setdefault("name", "orchestrator")
        kwargs.setdefault("capabilities", ["planning", "execution"])
        super().__init__(**kwargs)
        self.task = task
        self.available_agents = available_agents or ["researcher", "coder", "reviewer"]
        self._topology_engine = topology_engine
        self.workflow: Workflow | None = None
        self._router = get_router()

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        announce(ORCHESTRATOR, "Initializing", {
            "Task": self.task[:50] if self.task else "None",
            "Available Agents": ", ".join(self.available_agents),
        })

        await self._router.initialize()

    async def work(self) -> Workflow:
        """
        Execute orchestration workflow.

        Returns:
            Workflow with results
        """
        if not self.task:
            say(ORCHESTRATOR, "No task specified")
            return Workflow(task="")

        say(ORCHESTRATOR, f"Orchestrating: {self.task}")
        self.workflow = Workflow(task=self.task)

        # Step 1: Analyze task and select topology
        think(ORCHESTRATOR, "Analyzing task requirements...")
        self.increment_turn()
        topology_type = await self._select_topology()
        self.workflow.topology = topology_type
        say(ORCHESTRATOR, f"Selected topology: {topology_type}")

        # Step 2: Decompose into subtasks
        think(ORCHESTRATOR, "Decomposing task into subtasks...")
        self.increment_turn()
        subtasks = await self._decompose_task()
        self.workflow.subtasks = subtasks
        say(ORCHESTRATOR, f"Created {len(subtasks)} subtasks")

        # Step 3: Execute workflow based on topology
        think(ORCHESTRATOR, "Executing workflow...")
        await self._execute_workflow()

        # Step 4: Aggregate results
        think(ORCHESTRATOR, "Aggregating results...")
        self.increment_turn()
        final_result = await self._aggregate_results()
        self.workflow.results["final"] = final_result
        self.workflow.status = "completed"

        # Log decision
        await self.log_decision(
            decision=f"Orchestrated {len(subtasks)} subtasks with {topology_type} topology",
            category="orchestration",
            rationale=f"Task: {self.task[:100]}",
            tags=["orchestration", topology_type],
        )

        say(ORCHESTRATOR, "Orchestration complete")
        return self.workflow

    async def shutdown(self) -> None:
        """Cleanup orchestrator."""
        say(ORCHESTRATOR, "Orchestrator shutting down")

        # Store workflow pattern
        if self._hive_mind and self.workflow:
            await self.remember(
                content=f"Workflow for {self.task}:\n"
                f"Topology: {self.workflow.topology}\n"
                f"Subtasks: {len(self.workflow.subtasks)}",
                importance=0.7,
                tags=["workflow", "orchestration"],
            )

    async def _select_topology(self) -> str:
        """Select appropriate topology for the task."""
        if self._topology_engine:
            suggestion = self._topology_engine.suggest_topology(self.task)
            return suggestion["recommended"]

        # Simple heuristic if no engine
        task_lower = self.task.lower()

        if any(kw in task_lower for kw in ["consensus", "agree", "brainstorm"]):
            return "swarm"
        elif any(kw in task_lower for kw in ["review", "then", "after"]):
            return "pipeline"
        elif any(kw in task_lower for kw in ["coordinate", "manage"]):
            return "hierarchy"
        else:
            return "pipeline"  # Default

    async def _decompose_task(self) -> list[Subtask]:
        """Decompose task into subtasks."""
        messages = [
            LLMMessage(
                role="user",
                content=f"""Decompose this task into subtasks for a multi-agent system:

Task: {self.task}

Available agent types: {', '.join(self.available_agents)}

For each subtask, specify:
1. Description
2. Which agent type should handle it
3. Dependencies on other subtasks (if any)

Format:
SUBTASK: <description>
AGENT: <agent_type>
DEPENDS: <subtask_ids or "none">
---

Create 2-5 subtasks.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a task planner. Break down complex tasks efficiently.",
            max_tokens=800,
        )

        return self._parse_subtasks(response.content)

    def _parse_subtasks(self, content: str) -> list[Subtask]:
        """Parse subtasks from LLM response."""
        subtasks = []
        current: dict[str, Any] = {}
        subtask_id = 0

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("SUBTASK:"):
                if current:
                    subtasks.append(Subtask(
                        id=f"st-{subtask_id}",
                        description=current.get("description", ""),
                        agent_type=current.get("agent", "researcher"),
                        dependencies=current.get("depends", []),
                    ))
                    subtask_id += 1
                current = {"description": line[8:].strip()}

            elif line.startswith("AGENT:"):
                agent = line[6:].strip().lower()
                if agent in self.available_agents:
                    current["agent"] = agent
                else:
                    current["agent"] = "researcher"  # Fallback

            elif line.startswith("DEPENDS:"):
                deps = line[8:].strip().lower()
                if deps != "none":
                    current["depends"] = [d.strip() for d in deps.split(",")]
                else:
                    current["depends"] = []

        # Add last subtask
        if current:
            subtasks.append(Subtask(
                id=f"st-{subtask_id}",
                description=current.get("description", ""),
                agent_type=current.get("agent", "researcher"),
                dependencies=current.get("depends", []),
            ))

        return subtasks

    async def _execute_workflow(self) -> None:
        """Execute the workflow based on topology and execution mode."""
        if not self.workflow:
            return

        # Determine execution mode based on topology
        self.workflow.execution_mode = self._determine_execution_mode()
        say(ORCHESTRATOR, f"Execution mode: {self.workflow.execution_mode.value}")

        # Build dependency graph
        graph = DependencyGraph.from_subtasks(self.workflow.subtasks)
        self.workflow.execution_levels = graph.topological_sort()

        # Execute based on mode
        if self.workflow.execution_mode == ExecutionMode.PARALLEL:
            await self._execute_parallel(graph)
        elif self.workflow.execution_mode == ExecutionMode.STAGED:
            await self._execute_staged(graph)
        elif self.workflow.execution_mode == ExecutionMode.BROADCAST:
            await self._execute_broadcast(graph)
        else:
            await self._execute_sequential(graph)

    def _determine_execution_mode(self) -> ExecutionMode:
        """Determine execution mode based on topology type."""
        topology = self.workflow.topology if self.workflow else "pipeline"

        mode_map = {
            "swarm": ExecutionMode.BROADCAST,
            "pipeline": ExecutionMode.STAGED,
            "hierarchy": ExecutionMode.STAGED,
            "mesh": ExecutionMode.PARALLEL,
            "ring": ExecutionMode.SEQUENTIAL,
            "star": ExecutionMode.STAGED,
        }

        return mode_map.get(topology, ExecutionMode.SEQUENTIAL)

    async def _execute_sequential(self, graph: DependencyGraph) -> None:
        """Execute subtasks sequentially in dependency order."""
        completed: set[str] = set()

        while len(completed) < len(self.workflow.subtasks):
            ready = graph.get_ready_tasks(completed)
            if not ready:
                say(ORCHESTRATOR, "No more ready tasks (possible cycle or completion)")
                break

            # Execute one task at a time
            subtask = ready[0]
            await self._execute_subtask(subtask)
            completed.add(subtask.id)

    async def _execute_parallel(self, graph: DependencyGraph) -> None:
        """Execute independent subtasks in parallel."""
        completed: set[str] = set()

        while len(completed) < len(self.workflow.subtasks):
            ready = graph.get_ready_tasks(completed)
            if not ready:
                break

            say(ORCHESTRATOR, f"Executing {len(ready)} subtasks in parallel")

            # Execute all ready tasks concurrently
            tasks = [self._execute_subtask(st) for st in ready]
            await asyncio.gather(*tasks, return_exceptions=True)

            for st in ready:
                completed.add(st.id)

    async def _execute_staged(self, graph: DependencyGraph) -> None:
        """Execute subtasks in stages (levels) based on dependencies."""
        for level_idx, level_ids in enumerate(self.workflow.execution_levels):
            say(ORCHESTRATOR, f"Executing stage {level_idx + 1}/{len(self.workflow.execution_levels)}")

            # Get subtasks for this level
            level_subtasks = [graph.subtasks[st_id] for st_id in level_ids]

            if len(level_subtasks) == 1:
                # Single task, execute directly
                await self._execute_subtask(level_subtasks[0])
            else:
                # Multiple tasks in stage, execute in parallel
                tasks = [self._execute_subtask(st) for st in level_subtasks]
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_broadcast(self, graph: DependencyGraph) -> None:
        """Execute using swarm/broadcast pattern - all agents collaborate."""
        # In broadcast mode, we send the task context to all available agents
        # and let them work together, aggregating their responses

        say(ORCHESTRATOR, "Broadcasting to all agents (swarm mode)")

        # Group subtasks by agent type for specialized handling
        by_agent_type: dict[str, list[Subtask]] = {}
        for subtask in self.workflow.subtasks:
            by_agent_type.setdefault(subtask.agent_type, []).append(subtask)

        # Execute each agent type's tasks in parallel
        all_tasks = []
        for agent_type, subtasks in by_agent_type.items():
            for subtask in subtasks:
                # Include context from other agent types' tasks
                subtask.metadata = {"swarm_context": self._get_swarm_context(agent_type)}
                all_tasks.append(self._execute_subtask(subtask))

        await asyncio.gather(*all_tasks, return_exceptions=True)

    def _get_swarm_context(self, current_agent_type: str) -> str:
        """Get context from other agents for swarm collaboration."""
        if not self.workflow:
            return ""

        context_parts = []
        for subtask in self.workflow.subtasks:
            if subtask.agent_type != current_agent_type and subtask.result:
                context_parts.append(
                    f"[{subtask.agent_type}] {subtask.description}: {str(subtask.result)[:200]}"
                )

        return "\n".join(context_parts) if context_parts else "No prior results available."

    async def _execute_subtask(self, subtask: Subtask) -> None:
        """Execute a single subtask."""
        say(ORCHESTRATOR, f"Executing: {subtask.description[:40]}...")
        self.increment_turn()

        subtask.status = "running"

        try:
            # Route to appropriate agent based on topology
            result = await self._route_to_agent(subtask)
            subtask.result = result
            subtask.status = "completed"
            self.workflow.results[subtask.id] = result
            say(ORCHESTRATOR, f"Subtask {subtask.id} completed")

        except Exception as e:
            subtask.status = "failed"
            subtask.result = f"Error: {e}"
            self.workflow.results[subtask.id] = subtask.result
            logger.error(f"Subtask {subtask.id} failed: {e}")
            say(ORCHESTRATOR, f"Subtask {subtask.id} failed: {e}")

    async def _route_to_agent(self, subtask: Subtask) -> str:
        """Route subtask to appropriate agent based on topology and capabilities."""
        # Get topology info if available
        topology_role = None
        if self._topology_engine and self._topology_engine.current_topology:
            topology = self._topology_engine.current_topology
            # Find best agent for this task based on capabilities
            best_agent = self._find_best_agent(subtask, topology)
            if best_agent:
                subtask.assigned_agent_id = best_agent.agent_id
                topology_role = best_agent.role

        # For now, simulate agent execution
        # In a full implementation, this would spawn/delegate to actual agents
        return await self._simulate_agent(subtask, topology_role)

    def _find_best_agent(self, subtask: Subtask, topology: Any) -> Any:
        """Find the best agent in the topology for this subtask."""
        from hive.topology import AgentNode

        agents = topology.list_agents()
        if not agents:
            return None

        # Score agents based on capability match
        scored: list[tuple[float, AgentNode]] = []
        for agent in agents:
            score = 0.0

            # Match agent type
            if subtask.agent_type.lower() in agent.name.lower():
                score += 10.0

            # Match capabilities
            for cap in agent.capabilities:
                if cap.lower() in subtask.description.lower():
                    score += 2.0

            # Prefer specific roles for certain topologies
            if agent.role == "hub" and subtask.priority > 5:
                score += 5.0  # High priority to hub
            elif agent.role == "root":
                score += 3.0

            scored.append((score, agent))

        if not scored:
            return None

        # Return highest scoring agent
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    async def _simulate_agent(
        self,
        subtask: Subtask,
        topology_role: str | None = None,
    ) -> str:
        """Simulate agent execution (placeholder)."""
        # Build context based on dependencies
        dep_context = ""
        if subtask.dependencies and self.workflow:
            dep_results = []
            for dep_id in subtask.dependencies:
                if dep_id in self.workflow.results:
                    dep_results.append(
                        f"[{dep_id}]: {str(self.workflow.results[dep_id])[:200]}"
                    )
            if dep_results:
                dep_context = f"\n\nPrior results from dependencies:\n" + "\n".join(dep_results)

        # Build swarm context if available
        swarm_context = ""
        if hasattr(subtask, 'metadata') and subtask.metadata:
            if 'swarm_context' in subtask.metadata:
                swarm_context = f"\n\nContext from other agents:\n{subtask.metadata['swarm_context']}"

        # Build role context
        role_context = ""
        if topology_role:
            role_context = f"\nYour role in the topology: {topology_role}"

        messages = [
            LLMMessage(
                role="user",
                content=f"""You are a {subtask.agent_type} agent.{role_context} Complete this subtask:

{subtask.description}{dep_context}{swarm_context}

Provide a brief result (under 100 words).""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a {subtask.agent_type} agent completing a subtask.",
            max_tokens=200,
        )

        return response.content

    async def _aggregate_results(self) -> str:
        """Aggregate results from all subtasks."""
        if not self.workflow:
            return ""

        results_text = "\n\n".join(
            f"[{st.agent_type}] {st.description}:\n{st.result}"
            for st in self.workflow.subtasks
            if st.result
        )

        messages = [
            LLMMessage(
                role="user",
                content=f"""Aggregate these subtask results into a final response:

Original Task: {self.workflow.task}

Subtask Results:
{results_text}

Provide a coherent final answer that addresses the original task.""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a synthesizer. Combine results into a coherent response.",
            max_tokens=500,
        )

        return response.content
