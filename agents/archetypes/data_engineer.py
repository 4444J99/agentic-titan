"""
Data Engineer Agent - ETL, data quality, and schema management specialist.

Capabilities:
- ETL pipeline design
- Data quality validation
- Schema analysis and migration
- Query optimization
- Data lineage tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from adapters.base import LLMMessage
from adapters.router import get_router
from agents.framework.base_agent import BaseAgent

logger = logging.getLogger("titan.agents.data_engineer")


class DataFormat(Enum):
    """Supported data formats."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    ORC = "orc"
    DELTA = "delta"
    ICEBERG = "iceberg"


class DatabaseType(Enum):
    """Supported database types."""

    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    DATABRICKS = "databricks"
    MONGODB = "mongodb"


class QualityRuleType(Enum):
    """Types of data quality rules."""

    NOT_NULL = "not_null"
    UNIQUE = "unique"
    RANGE = "range"
    PATTERN = "pattern"
    REFERENTIAL = "referential"
    COMPLETENESS = "completeness"
    FRESHNESS = "freshness"
    CUSTOM = "custom"


@dataclass
class Column:
    """Schema column definition."""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: str | None = None
    default: Any = None
    description: str = ""


@dataclass
class TableSchema:
    """Database table schema."""

    name: str
    columns: list[Column] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    indexes: list[dict[str, Any]] = field(default_factory=list)
    partitions: list[str] = field(default_factory=list)


@dataclass
class QualityRule:
    """Data quality validation rule."""

    name: str
    rule_type: QualityRuleType
    column: str
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warning, info
    description: str = ""


@dataclass
class QualityResult:
    """Result of a data quality check."""

    rule_name: str
    passed: bool
    total_records: int = 0
    failed_records: int = 0
    failure_rate: float = 0.0
    sample_failures: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass
class ETLPipeline:
    """ETL pipeline definition."""

    name: str
    source: dict[str, Any]
    destination: dict[str, Any]
    transformations: list[dict[str, Any]] = field(default_factory=list)
    schedule: str = ""  # cron expression
    dependencies: list[str] = field(default_factory=list)
    quality_checks: list[QualityRule] = field(default_factory=list)


@dataclass
class QueryOptimization:
    """Query optimization result."""

    original_query: str
    optimized_query: str
    improvements: list[str] = field(default_factory=list)
    estimated_speedup: float = 1.0
    index_suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataEngineerAgent(BaseAgent):
    """
    Agent specialized in data engineering tasks.

    Capabilities:
    - Design ETL/ELT pipelines
    - Validate data quality
    - Analyze and migrate schemas
    - Optimize SQL queries
    - Track data lineage
    """

    def __init__(
        self,
        default_database: DatabaseType = DatabaseType.POSTGRESQL,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("name", "data_engineer")
        kwargs.setdefault(
            "capabilities",
            [
                "etl_design",
                "data_quality",
                "schema_analysis",
                "query_optimization",
                "data_lineage",
            ],
        )
        super().__init__(**kwargs)

        self.default_database = default_database
        self._router = get_router()
        self._pipelines: list[ETLPipeline] = []
        self._schemas: dict[str, TableSchema] = {}

    async def initialize(self) -> None:
        """Initialize the Data Engineer agent."""
        logger.info(f"Data Engineer Agent '{self.name}' initializing")
        await self._router.initialize()

    async def work(self) -> dict[str, Any]:
        """
        Main work loop - Data Engineer is typically invoked for specific tasks.

        Returns:
            Summary of capabilities and current state
        """
        logger.info("Data Engineer Agent ready for data engineering tasks")

        return {
            "status": "ready",
            "default_database": self.default_database.value,
            "pipelines_designed": len(self._pipelines),
            "schemas_analyzed": len(self._schemas),
        }

    async def shutdown(self) -> None:
        """Cleanup Data Engineer agent."""
        logger.info("Data Engineer Agent shutting down")

        if self._hive_mind and self._pipelines:
            summary = f"ETL pipelines designed: {len(self._pipelines)}"
            await self.remember(
                content=summary,
                importance=0.7,
                tags=["data_engineering", "etl", "pipelines"],
            )

    async def design_etl_pipeline(
        self,
        source_description: str,
        destination_description: str,
        requirements: list[str] | None = None,
        framework: str = "airflow",
    ) -> ETLPipeline:
        """
        Design an ETL/ELT pipeline.

        Args:
            source_description: Description of data source
            destination_description: Description of destination
            requirements: Business requirements
            framework: Target framework (airflow, dagster, prefect, dbt)

        Returns:
            ETLPipeline definition
        """
        self.increment_turn()

        messages = [
            LLMMessage(
                role="user",
                content=f"""Design an ETL pipeline for {framework}:

Source: {source_description}
Destination: {destination_description}
Requirements: {", ".join(requirements or ["standard ETL"])}

Provide:
NAME: pipeline_name
SOURCE:
- type: source_type
- connection: connection_details
DESTINATION:
- type: dest_type
- connection: connection_details
TRANSFORMATIONS:
- step: description
SCHEDULE: cron_expression
QUALITY_CHECKS:
- rule: description""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a data engineer expert in {framework}. Design robust ETL pipelines.",
            max_tokens=800,
        )

        pipeline = self._parse_pipeline(response.content)
        self._pipelines.append(pipeline)

        await self.log_decision(
            decision=f"Designed ETL pipeline: {pipeline.name}",
            category="etl_design",
            rationale=f"Source: {source_description[:50]}, Dest: {destination_description[:50]}",
            tags=["data_engineering", "etl", framework],
        )

        return pipeline

    def _parse_pipeline(self, content: str) -> ETLPipeline:
        """Parse pipeline definition from LLM response."""
        lines = content.split("\n")
        pipeline = ETLPipeline(
            name="unnamed_pipeline",
            source={},
            destination={},
        )

        current_section = None
        current_dict: dict[str, Any] | None = None

        for line in lines:
            line = line.strip()

            if line.startswith("NAME:"):
                pipeline.name = line.replace("NAME:", "").strip()
            elif line.startswith("SOURCE:"):
                current_section = "source"
                current_dict = pipeline.source
            elif line.startswith("DESTINATION:"):
                current_section = "destination"
                current_dict = pipeline.destination
            elif line.startswith("TRANSFORMATIONS:"):
                current_section = "transformations"
            elif line.startswith("SCHEDULE:"):
                pipeline.schedule = line.replace("SCHEDULE:", "").strip()
            elif line.startswith("QUALITY_CHECKS:"):
                current_section = "quality"
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section in ("source", "destination") and current_dict is not None:
                    if ":" in item:
                        key, val = item.split(":", 1)
                        current_dict[key.strip()] = val.strip()
                elif current_section == "transformations":
                    if ":" in item:
                        _, desc = item.split(":", 1)
                        pipeline.transformations.append({"step": desc.strip()})
                    else:
                        pipeline.transformations.append({"step": item})
                elif current_section == "quality":
                    if ":" in item:
                        _, desc = item.split(":", 1)
                        pipeline.quality_checks.append(
                            QualityRule(
                                name=f"check_{len(pipeline.quality_checks)}",
                                rule_type=QualityRuleType.CUSTOM,
                                column="*",
                                description=desc.strip(),
                            )
                        )

        return pipeline

    async def analyze_schema(
        self,
        ddl_or_description: str,
        database_type: DatabaseType | None = None,
    ) -> TableSchema:
        """
        Analyze a database schema from DDL or description.

        Args:
            ddl_or_description: DDL statement or schema description
            database_type: Target database type

        Returns:
            Parsed TableSchema
        """
        self.increment_turn()

        db_type = database_type or self.default_database

        messages = [
            LLMMessage(
                role="user",
                content=f"""Analyze this schema for {db_type.value}:

{ddl_or_description}

Provide structured output:
TABLE: table_name
COLUMNS:
- name: column_name, type: data_type, nullable: yes/no, pk: yes/no
INDEXES:
- columns: col1, col2, type: btree/hash
PARTITIONS:
- column: partition_column, strategy: range/list/hash""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a {db_type.value} database expert. Analyze schemas thoroughly.",
            max_tokens=600,
        )

        schema = self._parse_schema(response.content)
        self._schemas[schema.name] = schema

        return schema

    def _parse_schema(self, content: str) -> TableSchema:
        """Parse schema from LLM response."""
        schema = TableSchema(name="unknown")
        current_section = None

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("TABLE:"):
                schema.name = line.replace("TABLE:", "").strip()
            elif line.startswith("COLUMNS:"):
                current_section = "columns"
            elif line.startswith("INDEXES:"):
                current_section = "indexes"
            elif line.startswith("PARTITIONS:"):
                current_section = "partitions"
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "columns":
                    col = self._parse_column(item)
                    if col:
                        schema.columns.append(col)
                        if col.primary_key:
                            schema.primary_keys.append(col.name)
                elif current_section == "indexes":
                    schema.indexes.append({"definition": item})
                elif current_section == "partitions":
                    schema.partitions.append(item)

        return schema

    def _parse_column(self, col_str: str) -> Column | None:
        """Parse column definition."""
        # Format: name: col_name, type: data_type, nullable: yes/no, pk: yes/no
        parts = {}
        for part in col_str.split(","):
            if ":" in part:
                key, val = part.split(":", 1)
                parts[key.strip().lower()] = val.strip()

        if "name" not in parts:
            return None

        return Column(
            name=parts["name"],
            data_type=parts.get("type", "varchar"),
            nullable=parts.get("nullable", "yes").lower() in ("yes", "true", "1"),
            primary_key=parts.get("pk", "no").lower() in ("yes", "true", "1"),
        )

    async def generate_migration(
        self,
        source_schema: TableSchema,
        target_schema: TableSchema,
        database_type: DatabaseType | None = None,
    ) -> str:
        """
        Generate migration SQL from source to target schema.

        Args:
            source_schema: Current schema
            target_schema: Desired schema
            database_type: Target database

        Returns:
            Migration SQL statements
        """
        self.increment_turn()

        db_type = database_type or self.default_database

        # Build schema descriptions
        source_desc = self._schema_to_text(source_schema)
        target_desc = self._schema_to_text(target_schema)

        messages = [
            LLMMessage(
                role="user",
                content=f"""Generate migration SQL for {db_type.value}:

CURRENT SCHEMA:
{source_desc}

TARGET SCHEMA:
{target_desc}

Provide:
1. ALTER statements for column changes
2. CREATE INDEX statements
3. Data migration if needed
4. Rollback statements""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a {db_type.value} migration expert. Generate safe migrations.",
            max_tokens=800,
        )

        return response.content

    def _schema_to_text(self, schema: TableSchema) -> str:
        """Convert schema to text description."""
        lines = [f"Table: {schema.name}"]
        for col in schema.columns:
            pk = " (PK)" if col.primary_key else ""
            null = " NOT NULL" if not col.nullable else ""
            lines.append(f"  - {col.name}: {col.data_type}{null}{pk}")
        return "\n".join(lines)

    async def validate_data_quality(
        self,
        data_description: str,
        rules: list[QualityRule],
    ) -> list[QualityResult]:
        """
        Validate data quality against rules.

        Args:
            data_description: Description of data to validate
            rules: Quality rules to check

        Returns:
            List of validation results
        """
        self.increment_turn()

        rules_text = "\n".join(
            f"- {r.name} ({r.rule_type.value}): {r.description or r.column}" for r in rules
        )

        messages = [
            LLMMessage(
                role="user",
                content=f"""Simulate data quality validation:

DATA:
{data_description}

RULES:
{rules_text}

For each rule, provide:
RULE: rule_name
PASSED: yes/no
TOTAL: total_records
FAILED: failed_count
MESSAGE: explanation""",
            )
        ]

        response = await self._router.complete(
            messages,
            system="You are a data quality expert. Validate data thoroughly.",
            max_tokens=600,
        )

        return self._parse_quality_results(response.content, rules)

    def _parse_quality_results(
        self,
        content: str,
        rules: list[QualityRule],
    ) -> list[QualityResult]:
        """Parse quality validation results."""
        results = []
        current: dict[str, Any] = {}

        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("RULE:"):
                if current:
                    results.append(self._create_quality_result(current))
                current = {"rule": line.replace("RULE:", "").strip()}
            elif line.startswith("PASSED:") and current:
                current["passed"] = line.replace("PASSED:", "").strip().lower() in (
                    "yes",
                    "true",
                    "1",
                )
            elif line.startswith("TOTAL:") and current:
                try:
                    current["total"] = int(line.replace("TOTAL:", "").strip())
                except ValueError:
                    current["total"] = 0
            elif line.startswith("FAILED:") and current:
                try:
                    current["failed"] = int(line.replace("FAILED:", "").strip())
                except ValueError:
                    current["failed"] = 0
            elif line.startswith("MESSAGE:") and current:
                current["message"] = line.replace("MESSAGE:", "").strip()

        if current:
            results.append(self._create_quality_result(current))

        return results

    def _create_quality_result(self, data: dict[str, Any]) -> QualityResult:
        """Create QualityResult from parsed data."""
        total = data.get("total", 100)
        failed = data.get("failed", 0)

        return QualityResult(
            rule_name=data.get("rule", "unknown"),
            passed=data.get("passed", True),
            total_records=total,
            failed_records=failed,
            failure_rate=failed / max(total, 1),
            message=data.get("message", ""),
        )

    async def optimize_query(
        self,
        query: str,
        database_type: DatabaseType | None = None,
        table_schemas: list[TableSchema] | None = None,
    ) -> QueryOptimization:
        """
        Optimize a SQL query.

        Args:
            query: SQL query to optimize
            database_type: Target database
            table_schemas: Relevant table schemas

        Returns:
            QueryOptimization with suggestions
        """
        self.increment_turn()

        db_type = database_type or self.default_database

        schema_context = ""
        if table_schemas:
            schema_context = "SCHEMAS:\n" + "\n".join(
                self._schema_to_text(s) for s in table_schemas
            )

        messages = [
            LLMMessage(
                role="user",
                content=f"""Optimize this {db_type.value} query:

```sql
{query}
```

{schema_context}

Provide:
OPTIMIZED:
```sql
optimized_query
```
IMPROVEMENTS:
- improvement_description
SPEEDUP: estimated_factor (e.g., 2x, 5x)
INDEXES:
- suggested_index
WARNINGS:
- potential_issue""",
            )
        ]

        response = await self._router.complete(
            messages,
            system=f"You are a {db_type.value} query optimization expert.",
            max_tokens=800,
        )

        return self._parse_optimization(query, response.content)

    def _parse_optimization(self, original: str, content: str) -> QueryOptimization:
        """Parse query optimization result."""
        result = QueryOptimization(
            original_query=original,
            optimized_query=original,
        )

        # Extract optimized query
        if "```sql" in content.lower():
            parts = content.split("```sql", 1)
            if len(parts) > 1:
                sql_part = parts[1].split("```")[0]
                result.optimized_query = sql_part.strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                result.optimized_query = parts[1].strip()

        current_section = None
        for line in content.split("\n"):
            line = line.strip()

            if line.startswith("IMPROVEMENTS:"):
                current_section = "improvements"
            elif line.startswith("SPEEDUP:"):
                speedup_str = line.replace("SPEEDUP:", "").strip()
                try:
                    # Parse "2x", "5x", etc.
                    result.estimated_speedup = float(speedup_str.lower().replace("x", ""))
                except ValueError:
                    result.estimated_speedup = 1.0
            elif line.startswith("INDEXES:"):
                current_section = "indexes"
            elif line.startswith("WARNINGS:"):
                current_section = "warnings"
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "improvements":
                    result.improvements.append(item)
                elif current_section == "indexes":
                    result.index_suggestions.append(item)
                elif current_section == "warnings":
                    result.warnings.append(item)

        return result

    def create_quality_rules(
        self,
        schema: TableSchema,
        strict: bool = False,
    ) -> list[QualityRule]:
        """
        Generate data quality rules from schema.

        Args:
            schema: Table schema to generate rules for
            strict: Whether to generate strict rules

        Returns:
            List of quality rules
        """
        rules = []

        for col in schema.columns:
            # Not null check for non-nullable columns
            if not col.nullable:
                rules.append(
                    QualityRule(
                        name=f"{col.name}_not_null",
                        rule_type=QualityRuleType.NOT_NULL,
                        column=col.name,
                        severity="error",
                        description=f"{col.name} must not be null",
                    )
                )

            # Primary key uniqueness
            if col.primary_key:
                rules.append(
                    QualityRule(
                        name=f"{col.name}_unique",
                        rule_type=QualityRuleType.UNIQUE,
                        column=col.name,
                        severity="error",
                        description=f"{col.name} must be unique",
                    )
                )

            # Type-specific rules
            col_type = col.data_type.lower()
            if strict:
                if "int" in col_type or "numeric" in col_type or "decimal" in col_type:
                    rules.append(
                        QualityRule(
                            name=f"{col.name}_range",
                            rule_type=QualityRuleType.RANGE,
                            column=col.name,
                            parameters={"min": 0},  # Adjust based on context
                            severity="warning",
                            description=f"{col.name} should be non-negative",
                        )
                    )

                if "email" in col.name.lower():
                    rules.append(
                        QualityRule(
                            name=f"{col.name}_email_format",
                            rule_type=QualityRuleType.PATTERN,
                            column=col.name,
                            parameters={"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
                            severity="error",
                            description=f"{col.name} must be valid email",
                        )
                    )

        return rules

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Get summary of designed pipelines."""
        if not self._pipelines:
            return {"count": 0}

        return {
            "count": len(self._pipelines),
            "pipelines": [
                {
                    "name": p.name,
                    "source_type": p.source.get("type", "unknown"),
                    "dest_type": p.destination.get("type", "unknown"),
                    "transformations": len(p.transformations),
                    "quality_checks": len(p.quality_checks),
                }
                for p in self._pipelines
            ],
        }
