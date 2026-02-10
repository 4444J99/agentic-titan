"""Tests for visualization generator module."""

from __future__ import annotations

from datetime import datetime

import pytest

from titan.workflows.inquiry_config import QUICK_INQUIRY_WORKFLOW
from titan.workflows.inquiry_engine import InquirySession, InquiryStatus, StageResult
from titan.workflows.visualization_generator import (
    VisualizationGenerator,
    VisualizationLibrary,
    VisualizationSpec,
    VisualizationSuite,
    VisualizationType,
    generate_visualizations,
    get_visualization_generator,
)


class TestVisualizationType:
    """Tests for VisualizationType enum."""

    def test_visualization_type_values(self) -> None:
        """Test all visualization types have correct values."""
        assert VisualizationType.RADAR.value == "radar"
        assert VisualizationType.BAR.value == "bar"
        assert VisualizationType.SANKEY.value == "sankey"
        assert VisualizationType.FORCE.value == "force"
        assert VisualizationType.TREEMAP.value == "treemap"
        assert VisualizationType.PIE.value == "pie"
        assert VisualizationType.LINE.value == "line"


class TestVisualizationLibrary:
    """Tests for VisualizationLibrary enum."""

    def test_library_values(self) -> None:
        """Test library values."""
        assert VisualizationLibrary.CHARTJS.value == "chartjs"
        assert VisualizationLibrary.D3.value == "d3"
        assert VisualizationLibrary.PLOTLY.value == "plotly"


class TestVisualizationSpec:
    """Tests for VisualizationSpec dataclass."""

    def test_create_spec(self) -> None:
        """Test creating a visualization spec."""
        spec = VisualizationSpec(
            viz_type=VisualizationType.RADAR,
            library=VisualizationLibrary.CHARTJS,
            title="Test Radar",
            description="A test radar chart",
            config={"responsive": True},
            data={"labels": ["A", "B", "C"], "values": [1, 2, 3]},
        )

        assert spec.viz_type == VisualizationType.RADAR
        assert spec.library == VisualizationLibrary.CHARTJS
        assert spec.title == "Test Radar"

    def test_default_dimensions(self) -> None:
        """Test default width and height."""
        spec = VisualizationSpec(
            viz_type=VisualizationType.BAR,
            library=VisualizationLibrary.CHARTJS,
            title="Test",
            description="Test",
            config={},
            data={},
        )

        assert spec.width == 600
        assert spec.height == 400
        assert spec.responsive is True

    def test_to_dict(self) -> None:
        """Test converting spec to dict."""
        spec = VisualizationSpec(
            viz_type=VisualizationType.PIE,
            library=VisualizationLibrary.CHARTJS,
            title="Pie Chart",
            description="Distribution",
            config={"plugins": {}},
            data={"labels": ["X", "Y"], "datasets": []},
        )

        d = spec.to_dict()

        assert d["viz_type"] == "pie"
        assert d["library"] == "chartjs"
        assert d["title"] == "Pie Chart"
        assert "config" in d
        assert "data" in d

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        spec = VisualizationSpec(
            viz_type=VisualizationType.LINE,
            library=VisualizationLibrary.CHARTJS,
            title="Line",
            description="Line chart",
            config={},
            data={},
        )

        json_str = spec.to_json()

        assert '"viz_type": "line"' in json_str
        assert '"library": "chartjs"' in json_str

    def test_to_chartjs_config(self) -> None:
        """Test Chart.js config generation."""
        spec = VisualizationSpec(
            viz_type=VisualizationType.BAR,
            library=VisualizationLibrary.CHARTJS,
            title="Bar Chart",
            description="Test",
            config={"scales": {}},
            data={"labels": ["A"], "datasets": []},
        )

        chartjs = spec.to_chartjs_config()

        assert chartjs["type"] == "bar"
        assert chartjs["data"] == spec.data
        assert chartjs["options"]["responsive"] is True
        assert chartjs["options"]["plugins"]["title"]["text"] == "Bar Chart"

    def test_to_chartjs_config_wrong_library(self) -> None:
        """Test Chart.js config fails for non-Chart.js spec."""
        spec = VisualizationSpec(
            viz_type=VisualizationType.FORCE,
            library=VisualizationLibrary.D3,
            title="Force Graph",
            description="D3 visualization",
            config={},
            data={},
        )

        with pytest.raises(ValueError, match="Not a Chart.js visualization"):
            spec.to_chartjs_config()


class TestVisualizationSuite:
    """Tests for VisualizationSuite dataclass."""

    def test_create_suite(self) -> None:
        """Test creating a visualization suite."""
        spec1 = VisualizationSpec(
            viz_type=VisualizationType.RADAR,
            library=VisualizationLibrary.CHARTJS,
            title="Radar",
            description="Test",
            config={},
            data={},
        )
        spec2 = VisualizationSpec(
            viz_type=VisualizationType.BAR,
            library=VisualizationLibrary.CHARTJS,
            title="Bar",
            description="Test",
            config={},
            data={},
        )

        suite = VisualizationSuite(
            session_id="inq-123",
            topic="Test Topic",
            visualizations=[spec1, spec2],
        )

        assert suite.session_id == "inq-123"
        assert len(suite.visualizations) == 2

    def test_to_dict(self) -> None:
        """Test converting suite to dict."""
        suite = VisualizationSuite(
            session_id="inq-test",
            topic="Topic",
            visualizations=[],
            recommended_layout="tabs",
        )

        d = suite.to_dict()

        assert d["session_id"] == "inq-test"
        assert d["recommended_layout"] == "tabs"

    def test_get_by_type(self) -> None:
        """Test filtering visualizations by type."""
        radar = VisualizationSpec(
            viz_type=VisualizationType.RADAR,
            library=VisualizationLibrary.CHARTJS,
            title="R1",
            description="",
            config={},
            data={},
        )
        bar = VisualizationSpec(
            viz_type=VisualizationType.BAR,
            library=VisualizationLibrary.CHARTJS,
            title="B1",
            description="",
            config={},
            data={},
        )

        suite = VisualizationSuite(
            session_id="test",
            topic="Test",
            visualizations=[radar, bar],
        )

        radars = suite.get_by_type(VisualizationType.RADAR)
        bars = suite.get_by_type(VisualizationType.BAR)

        assert len(radars) == 1
        assert len(bars) == 1


class TestVisualizationGenerator:
    """Tests for VisualizationGenerator class."""

    @pytest.fixture
    def generator(self) -> VisualizationGenerator:
        """Create a generator for testing."""
        return VisualizationGenerator()

    @pytest.fixture
    def mock_session(self) -> InquirySession:
        """Create a mock inquiry session."""
        session = InquirySession(
            id="inq-viz-test",
            topic="Data Visualization Techniques",
            workflow=QUICK_INQUIRY_WORKFLOW,
            status=InquiryStatus.COMPLETED,
        )

        session.results = [
            StageResult(
                stage_name="Scope Clarification",
                role="Scope AI",
                content=(
                    "Data visualization involves multiple techniques: bar charts, "
                    "line graphs, scatter plots."
                ),
                model_used="claude-3",
                timestamp=datetime.now(),
                duration_ms=1500,
                stage_index=0,
            ),
            StageResult(
                stage_name="Logical Analysis",
                role="Logic AI",
                content=(
                    "1. Charts help pattern recognition\n"
                    "2. Colors convey meaning\n"
                    "3. Interactivity improves engagement"
                ),
                model_used="gpt-4",
                timestamp=datetime.now(),
                duration_ms=2000,
                stage_index=1,
            ),
            StageResult(
                stage_name="Pattern Recognition",
                role="Pattern AI",
                content=(
                    "Key patterns: 30% use bar charts, 25% use line graphs, 45% use combinations."
                ),
                model_used="claude-3",
                timestamp=datetime.now(),
                duration_ms=1800,
                stage_index=2,
            ),
        ]

        return session

    @pytest.mark.asyncio
    async def test_generate_suite_basic(
        self, generator: VisualizationGenerator, mock_session: InquirySession
    ) -> None:
        """Test basic suite generation."""
        suite = await generator.generate_suite(mock_session)

        assert suite.session_id == mock_session.id
        assert suite.topic == mock_session.topic
        assert len(suite.visualizations) >= 2

    @pytest.mark.asyncio
    async def test_generates_stage_radar(
        self, generator: VisualizationGenerator, mock_session: InquirySession
    ) -> None:
        """Test that stage radar is generated."""
        suite = await generator.generate_suite(mock_session)

        radars = suite.get_by_type(VisualizationType.RADAR)
        assert len(radars) >= 1

    @pytest.mark.asyncio
    async def test_generates_style_distribution(
        self, generator: VisualizationGenerator, mock_session: InquirySession
    ) -> None:
        """Test that style distribution pie is generated."""
        suite = await generator.generate_suite(mock_session)

        pies = suite.get_by_type(VisualizationType.PIE)
        assert len(pies) >= 1

    @pytest.mark.asyncio
    async def test_generates_timeline(
        self, generator: VisualizationGenerator, mock_session: InquirySession
    ) -> None:
        """Test that timeline bar chart is generated."""
        suite = await generator.generate_suite(mock_session)

        bars = [v for v in suite.visualizations if "Timeline" in v.title]
        assert len(bars) >= 1

    @pytest.mark.asyncio
    async def test_generates_concept_graph(
        self, generator: VisualizationGenerator, mock_session: InquirySession
    ) -> None:
        """Test that concept graph is generated."""
        suite = await generator.generate_suite(mock_session)

        graphs = suite.get_by_type(VisualizationType.FORCE)
        assert len(graphs) >= 1

    def test_generate_stage_radar(
        self, generator: VisualizationGenerator, mock_session: InquirySession
    ) -> None:
        """Test stage radar generation."""
        radar = generator._generate_stage_radar(mock_session)

        assert radar.viz_type == VisualizationType.RADAR
        assert radar.library == VisualizationLibrary.CHARTJS
        assert "labels" in radar.data
        assert len(radar.data["labels"]) == len(mock_session.results)

    def test_generate_style_distribution(
        self, generator: VisualizationGenerator, mock_session: InquirySession
    ) -> None:
        """Test style distribution generation."""
        pie = generator._generate_style_distribution(mock_session)

        assert pie.viz_type == VisualizationType.PIE
        assert "labels" in pie.data
        assert "datasets" in pie.data

    def test_generate_timeline(
        self, generator: VisualizationGenerator, mock_session: InquirySession
    ) -> None:
        """Test timeline generation."""
        timeline = generator._generate_timeline(mock_session)

        assert timeline.viz_type == VisualizationType.BAR
        assert len(timeline.data["labels"]) == len(mock_session.results)
        # Check duration data
        assert timeline.data["datasets"][0]["data"][0] == 1.5  # 1500ms = 1.5s

    def test_parse_content_for_viz_with_list(self, generator: VisualizationGenerator) -> None:
        """Test parsing content with numbered list."""
        content = """Key points:
1. First important point
2. Second point about data
3. Third analytical insight
4. Fourth conclusion
"""

        viz = generator._parse_content_for_viz(content, "Test Stage")

        assert viz is not None
        assert viz.viz_type == VisualizationType.BAR
        assert len(viz.data["labels"]) == 4

    def test_parse_content_for_viz_with_percentages(
        self, generator: VisualizationGenerator
    ) -> None:
        """Test parsing content with percentages."""
        content = """
        Distribution analysis:
        Category A represents 35% of the data.
        Category B accounts for 28%.
        Category C makes up 37%.
        """

        viz = generator._parse_content_for_viz(content, "Analysis Stage")

        assert viz is not None
        assert viz.viz_type == VisualizationType.PIE
        assert 35.0 in viz.data["datasets"][0]["data"]

    def test_parse_content_for_viz_no_match(self, generator: VisualizationGenerator) -> None:
        """Test parsing content without extractable data."""
        content = "This is just plain text without any structured data."

        viz = generator._parse_content_for_viz(content, "Plain Stage")

        assert viz is None

    def test_generate_concept_graph(
        self, generator: VisualizationGenerator, mock_session: InquirySession
    ) -> None:
        """Test concept graph generation."""
        graph = generator._generate_concept_graph(mock_session)

        assert graph.viz_type == VisualizationType.FORCE
        assert graph.library == VisualizationLibrary.D3
        assert "nodes" in graph.data
        assert "links" in graph.data

        # Should have topic node plus stage nodes
        assert len(graph.data["nodes"]) >= len(mock_session.results) + 1

    def test_parse_visualization_output_with_json(self, generator: VisualizationGenerator) -> None:
        """Test parsing output with embedded JSON."""
        content = """
        Analysis results:

        ```json
        {
            "labels": ["A", "B", "C"],
            "values": [10, 20, 30]
        }
        ```

        Conclusion follows.
        """

        specs = generator.parse_visualization_output(content, "JSON Stage")

        # Should extract the JSON block
        json_specs = [s for s in specs if s.metadata.get("source") == "json_block"]
        assert len(json_specs) >= 1


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_visualization_generator_singleton(self) -> None:
        """Test singleton behavior."""
        gen1 = get_visualization_generator()
        gen2 = get_visualization_generator()

        assert gen1 is gen2

    @pytest.mark.asyncio
    async def test_generate_visualizations_function(self) -> None:
        """Test the generate_visualizations convenience function."""
        session = InquirySession(
            id="inq-conv-test",
            topic="Test",
            workflow=QUICK_INQUIRY_WORKFLOW,
        )
        session.results = [
            StageResult(
                stage_name="Stage",
                role="AI",
                content="Content",
                model_used="model",
                timestamp=datetime.now(),
                duration_ms=100,
                stage_index=0,
            ),
            StageResult(
                stage_name="Stage 2",
                role="AI 2",
                content="More content",
                model_used="model",
                timestamp=datetime.now(),
                duration_ms=200,
                stage_index=1,
            ),
        ]

        suite = await generate_visualizations(session)

        assert suite.session_id == session.id
        assert len(suite.visualizations) >= 1
