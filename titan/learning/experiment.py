"""
Titan Learning - Experiment Tracking

Integration with experiment tracking systems (Weights & Biases, MLflow, etc.)
for RLHF training runs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger("titan.learning.experiment")


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""

    # Project settings
    project_name: str = "titan-rlhf"
    entity: str | None = None  # W&B team/org

    # Run settings
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # Tracking options
    track_code: bool = True
    track_config: bool = True
    track_metrics: bool = True
    track_artifacts: bool = True

    # Backend selection
    backend: str = "wandb"  # wandb, mlflow, local

    # Local fallback
    local_dir: str = "./experiments"

    # W&B specific
    wandb_mode: str = "online"  # online, offline, disabled

    # MLflow specific
    mlflow_tracking_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_name": self.project_name,
            "entity": self.entity,
            "run_name": self.run_name,
            "tags": self.tags,
            "notes": self.notes,
            "track_code": self.track_code,
            "track_config": self.track_config,
            "track_metrics": self.track_metrics,
            "track_artifacts": self.track_artifacts,
            "backend": self.backend,
            "local_dir": self.local_dir,
            "wandb_mode": self.wandb_mode,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
        }


@dataclass
class ExperimentRun:
    """Represents a single experiment run."""

    run_id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    status: str = "running"  # running, completed, failed, aborted
    metrics: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "name": self.name,
            "config": self.config,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "metrics": {k: list(v) for k, v in self.metrics.items()},
            "artifacts": self.artifacts,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRun:
        """Create from dictionary."""
        run = cls(
            run_id=data.get("run_id", str(uuid4())[:8]),
            name=data.get("name", ""),
            config=data.get("config", {}),
            status=data.get("status", "running"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )
        if data.get("started_at"):
            run.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            run.completed_at = datetime.fromisoformat(data["completed_at"])
        if data.get("metrics"):
            run.metrics = {k: [(s, v) for s, v in v] for k, v in data["metrics"].items()}
        run.artifacts = data.get("artifacts", [])
        return run


class ExperimentTracker:
    """
    Experiment tracking for RLHF training.

    Supports multiple backends:
    - Weights & Biases (wandb)
    - MLflow
    - Local filesystem fallback
    """

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config or ExperimentConfig()
        self._current_run: ExperimentRun | None = None
        self._backend = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the tracking backend.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            if self._config.backend == "wandb":
                self._init_wandb()
            elif self._config.backend == "mlflow":
                self._init_mlflow()
            else:
                self._init_local()

            self._initialized = True
            logger.info(f"Experiment tracker initialized with backend: {self._config.backend}")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize {self._config.backend}, falling back to local: {e}")
            self._config.backend = "local"
            self._init_local()
            self._initialized = True
            return True

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases backend."""
        import wandb

        os.environ["WANDB_MODE"] = self._config.wandb_mode
        self._backend = wandb

    def _init_mlflow(self) -> None:
        """Initialize MLflow backend."""
        import mlflow

        if self._config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self._config.mlflow_tracking_uri)
        mlflow.set_experiment(self._config.project_name)
        self._backend = mlflow

    def _init_local(self) -> None:
        """Initialize local filesystem backend."""
        os.makedirs(self._config.local_dir, exist_ok=True)
        self._backend = "local"

    def start_run(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> ExperimentRun:
        """
        Start a new experiment run.

        Args:
            name: Run name
            config: Run configuration
            tags: Run tags

        Returns:
            ExperimentRun instance
        """
        if not self._initialized:
            self.initialize()

        run_name = name or self._config.run_name or f"run-{uuid4().hex[:8]}"
        run_config = config or {}
        run_tags = (tags or []) + self._config.tags

        run = ExperimentRun(
            name=run_name,
            config=run_config,
            tags=run_tags,
            notes=self._config.notes,
        )
        self._current_run = run

        try:
            if self._config.backend == "wandb":
                import wandb

                wandb.init(
                    project=self._config.project_name,
                    entity=self._config.entity,
                    name=run_name,
                    config=run_config,
                    tags=run_tags,
                    notes=self._config.notes,
                )
                run.run_id = wandb.run.id

            elif self._config.backend == "mlflow":
                import mlflow

                mlflow.start_run(run_name=run_name, tags={t: "true" for t in run_tags})
                mlflow.log_params(self._flatten_config(run_config))
                run.run_id = mlflow.active_run().info.run_id

            else:
                # Local backend
                run_dir = Path(self._config.local_dir) / run.run_id
                os.makedirs(run_dir, exist_ok=True)
                self._save_run_local(run)

            logger.info(f"Started experiment run: {run.run_id} ({run_name})")

        except Exception as e:
            logger.error(f"Error starting run: {e}")
            run.status = "failed"

        return run

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log metrics for the current run.

        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number
        """
        if not self._current_run:
            logger.warning("No active run, metrics not logged")
            return

        # Track in run object
        for name, value in metrics.items():
            if name not in self._current_run.metrics:
                self._current_run.metrics[name] = []
            self._current_run.metrics[name].append((step or 0, value))

        try:
            if self._config.backend == "wandb":
                import wandb

                wandb.log(metrics, step=step)

            elif self._config.backend == "mlflow":
                import mlflow

                mlflow.log_metrics(metrics, step=step)

            else:
                # Local backend - save to file
                self._save_run_local(self._current_run)

        except Exception as e:
            logger.warning(f"Error logging metrics: {e}")

    def log_config(self, config: dict[str, Any]) -> None:
        """
        Log configuration for the current run.

        Args:
            config: Configuration dictionary
        """
        if not self._current_run:
            logger.warning("No active run, config not logged")
            return

        self._current_run.config.update(config)

        try:
            if self._config.backend == "wandb":
                import wandb

                wandb.config.update(config)

            elif self._config.backend == "mlflow":
                import mlflow

                mlflow.log_params(self._flatten_config(config))

            else:
                self._save_run_local(self._current_run)

        except Exception as e:
            logger.warning(f"Error logging config: {e}")

    def log_artifact(
        self,
        path: str,
        name: str | None = None,
        artifact_type: str = "file",
    ) -> None:
        """
        Log an artifact for the current run.

        Args:
            path: Path to artifact
            name: Optional artifact name
            artifact_type: Type of artifact (file, model, dataset, etc.)
        """
        if not self._current_run:
            logger.warning("No active run, artifact not logged")
            return

        self._current_run.artifacts.append(path)

        try:
            if self._config.backend == "wandb":
                import wandb

                artifact = wandb.Artifact(
                    name=name or Path(path).name,
                    type=artifact_type,
                )
                artifact.add_file(path)
                wandb.log_artifact(artifact)

            elif self._config.backend == "mlflow":
                import mlflow

                mlflow.log_artifact(path)

            else:
                # Local backend - copy to run dir
                import shutil

                run_dir = Path(self._config.local_dir) / self._current_run.run_id
                artifacts_dir = run_dir / "artifacts"
                os.makedirs(artifacts_dir, exist_ok=True)
                shutil.copy(path, artifacts_dir)
                self._save_run_local(self._current_run)

        except Exception as e:
            logger.warning(f"Error logging artifact: {e}")

    def log_model(
        self,
        model_path: str,
        name: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a trained model.

        Args:
            model_path: Path to model directory
            name: Model name
            metadata: Optional model metadata
        """
        self.log_artifact(model_path, name=name, artifact_type="model")

        if metadata:
            self.log_config({"model_metadata": metadata})

    def end_run(self, status: str = "completed") -> None:
        """
        End the current run.

        Args:
            status: Final status (completed, failed, aborted)
        """
        if not self._current_run:
            return

        self._current_run.status = status
        self._current_run.completed_at = datetime.now(timezone.utc)

        try:
            if self._config.backend == "wandb":
                import wandb

                wandb.finish(exit_code=0 if status == "completed" else 1)

            elif self._config.backend == "mlflow":
                import mlflow

                mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")

            else:
                self._save_run_local(self._current_run)

            logger.info(f"Ended experiment run: {self._current_run.run_id} ({status})")

        except Exception as e:
            logger.warning(f"Error ending run: {e}")

        self._current_run = None

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """
        Get a previous run by ID.

        Args:
            run_id: Run ID

        Returns:
            ExperimentRun or None if not found
        """
        try:
            if self._config.backend == "wandb":
                import wandb

                api = wandb.Api()
                run = api.run(f"{self._config.entity}/{self._config.project_name}/{run_id}")
                return ExperimentRun(
                    run_id=run.id,
                    name=run.name,
                    config=dict(run.config),
                    status="completed" if run.state == "finished" else run.state,
                    tags=list(run.tags),
                )

            elif self._config.backend == "mlflow":
                import mlflow

                run = mlflow.get_run(run_id)
                return ExperimentRun(
                    run_id=run.info.run_id,
                    name=run.info.run_name or "",
                    config=dict(run.data.params),
                    status="completed" if run.info.status == "FINISHED" else run.info.status.lower(),
                )

            else:
                # Local backend
                run_dir = Path(self._config.local_dir) / run_id
                run_file = run_dir / "run.json"
                if run_file.exists():
                    with open(run_file) as f:
                        return ExperimentRun.from_dict(json.load(f))
                return None

        except Exception as e:
            logger.warning(f"Error getting run {run_id}: {e}")
            return None

    def list_runs(
        self,
        limit: int = 100,
        status: str | None = None,
    ) -> list[ExperimentRun]:
        """
        List experiment runs.

        Args:
            limit: Maximum number of runs to return
            status: Filter by status

        Returns:
            List of ExperimentRun
        """
        runs = []

        try:
            if self._config.backend == "wandb":
                import wandb

                api = wandb.Api()
                wandb_runs = api.runs(
                    f"{self._config.entity}/{self._config.project_name}",
                    per_page=limit,
                )
                for run in wandb_runs:
                    runs.append(
                        ExperimentRun(
                            run_id=run.id,
                            name=run.name,
                            config=dict(run.config),
                            status="completed" if run.state == "finished" else run.state,
                            tags=list(run.tags),
                        )
                    )

            elif self._config.backend == "mlflow":
                import mlflow

                experiment = mlflow.get_experiment_by_name(self._config.project_name)
                if experiment:
                    mlflow_runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        max_results=limit,
                    )
                    for _, row in mlflow_runs.iterrows():
                        runs.append(
                            ExperimentRun(
                                run_id=row["run_id"],
                                name=row.get("tags.mlflow.runName", ""),
                                status=(
                                    "completed"
                                    if row["status"] == "FINISHED"
                                    else row["status"].lower()
                                ),
                            )
                        )

            else:
                # Local backend
                local_dir = Path(self._config.local_dir)
                if local_dir.exists():
                    for run_dir in sorted(local_dir.iterdir(), reverse=True)[:limit]:
                        run_file = run_dir / "run.json"
                        if run_file.exists():
                            with open(run_file) as f:
                                runs.append(ExperimentRun.from_dict(json.load(f)))

        except Exception as e:
            logger.warning(f"Error listing runs: {e}")

        # Filter by status
        if status:
            runs = [r for r in runs if r.status == status]

        return runs

    def _flatten_config(self, config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested config for MLflow."""
        flat = {}
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, full_key))
            else:
                flat[full_key] = str(value)
        return flat

    def _save_run_local(self, run: ExperimentRun) -> None:
        """Save run to local filesystem."""
        run_dir = Path(self._config.local_dir) / run.run_id
        os.makedirs(run_dir, exist_ok=True)

        run_file = run_dir / "run.json"
        with open(run_file, "w") as f:
            json.dump(run.to_dict(), f, indent=2)

    @property
    def current_run(self) -> ExperimentRun | None:
        """Get current run."""
        return self._current_run


# Factory function
_tracker: ExperimentTracker | None = None


def get_experiment_tracker(config: ExperimentConfig | None = None) -> ExperimentTracker:
    """Get the default experiment tracker."""
    global _tracker
    if _tracker is None or config is not None:
        _tracker = ExperimentTracker(config)
    return _tracker
