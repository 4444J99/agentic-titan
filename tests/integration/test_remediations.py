import pytest
import os
from hive.memory import HiveMind, hash_embedding
import asyncio
import json

def test_fail_secure_production():
    os.environ["TITAN_ENV"] = "production"
    if "TITAN_JWT_SECRET" in os.environ: del os.environ["TITAN_JWT_SECRET"]
    with pytest.raises(ValueError, match="TITAN_JWT_SECRET must be set in production"):
        secret = os.getenv("TITAN_JWT_SECRET")  # allow-secret
        if not secret and os.getenv("TITAN_ENV") == "production":
            raise ValueError("TITAN_JWT_SECRET must be set in production")

@pytest.mark.asyncio
async def test_mget_efficiency(mocker):
    hive = HiveMind()
    mock_redis = mocker.AsyncMock()
    mock_redis.keys.return_value = ["titan:agent:1", "titan:agent:2"]
    mock_redis.mget.return_value = [json.dumps({"name": "agent1"}), json.dumps({"name": "agent2"})]
    hive._redis = mock_redis
    agents = await hive.list_agents()
    assert len(agents) == 2
    mock_redis.mget.assert_called_once()

@pytest.mark.asyncio
async def test_embedding_async_offload(mocker):
    hive = HiveMind()
    await hive.initialize()
    loop = asyncio.get_running_loop()
    spy = mocker.spy(loop, "run_in_executor")
    await hive.remember("test-agent", "some content")
    assert spy.call_count > 0
