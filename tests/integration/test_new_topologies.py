import pytest
from hive.topology import TopologyEngine, TopologyType, TaskProfile

def test_topology_creation():
    engine = TopologyEngine()
    
    # Test Rhizomatic
    topo = engine.create_topology(TopologyType.RHIZOMATIC)
    assert topo.topology_type == TopologyType.RHIZOMATIC
    
    # Test Fission-Fusion
    topo = engine.create_topology(TopologyType.FISSION_FUSION)
    assert topo.topology_type == TopologyType.FISSION_FUSION
    
    # Test Stigmergic
    topo = engine.create_topology(TopologyType.STIGMERGIC)
    assert topo.topology_type == TopologyType.STIGMERGIC

def test_topology_selection():
    engine = TopologyEngine()
    
    # Test lateral selection
    selected = engine.select_topology("A decentralized grassroots network task")
    assert selected == TopologyType.RHIZOMATIC
    
    # Test modular selection
    selected = engine.select_topology("A modular task with multiple sub-groups and fission events")
    assert selected == TopologyType.FISSION_FUSION
    
    # Test stigmergic selection
    selected = engine.select_topology("Environment-mediated coordination via pheromone trails")
    assert selected == TopologyType.STIGMERGIC

if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
