import pytest
from unittest.mock import MagicMock, AsyncMock
from titan.workflows.inquiry_engine import InquirySession, StageResult
from titan.workflows.quality_gates import DialecticGate, QualityGateResult

@pytest.mark.asyncio
async def test_dialectic_gate():
    # Mock session
    session = MagicMock(spec=InquirySession)
    session.id = "test-session"
    session.results = [
        StageResult(stage_name="Stage 1", role="Role A", content="Sky is blue.", model_used="gpt-4", timestamp=None),
        StageResult(stage_name="Stage 2", role="Role B", content="Sky is green.", model_used="gpt-4", timestamp=None)
    ]
    
    # Mock LLM caller
    mock_llm = AsyncMock(return_value='{"contradictions": ["Blue vs Green"], "tensions": [], "friction_score": 0.9}')
    
    gate = DialecticGate(llm_caller=mock_llm)
    result = await gate.evaluate(session)
    
    assert isinstance(result, QualityGateResult)
    assert not result.passed  # Should fail due to high friction
    assert result.score == pytest.approx(0.1)
    assert "Blue vs Green" in result.issues

if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
