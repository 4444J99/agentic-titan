"""
Titan Orchestration - Workflow Management and Termination

Provides:
- Formal termination conditions
- Execution watchdog
- Workflow lifecycle management
"""

from titan.orchestration.termination import (
    TerminationCondition,
    TerminationReason,
    TerminationResult,
    DefaultTerminationConditions,
)
from titan.orchestration.watchdog import (
    ExecutionWatchdog,
    WatchdogConfig,
    WatchdogAlert,
    get_watchdog,
)

__all__ = [
    # Termination
    "TerminationCondition",
    "TerminationReason",
    "TerminationResult",
    "DefaultTerminationConditions",
    # Watchdog
    "ExecutionWatchdog",
    "WatchdogConfig",
    "WatchdogAlert",
    "get_watchdog",
]
