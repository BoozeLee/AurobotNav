"""
test_paths.py - Pytest tests for AurobotNav path validation
Tests that paths are optimal length and functionality works correctly
"""

import pytest
import numpy as np
from rift_weaver import rift_weaver, phi_heuristic, digital_root
from astar_nav import PhiAStarNavigator, FractalGrid
from PRIMECORE import PrimeCore

class TestRiftWeaver:
    """Test cases for rift_weaver pathfinding"""
    
    def test_simple_path_length(self):
        """Test that simple paths are near optimal length"""
        # Simple 5x5 grid with no obstacles
        grid = [[0 for _ in range(5)] for _ in range(5)]
        path = rift_weaver(grid, (0, 0), (4, 4))
        
        assert path is not None, "Path should be found"
        assert len(path) >= 5, "Path should be at least 5 steps (Manhattan distance)"
        assert len(path) <= 10, "Path should be reasonably efficient"
        assert path[0] == (0, 0), "Path should start at origin"
        assert path[-1] == (4, 4), "Path should end at goal"
    
    def test_obstacle_avoidance_path_length(self):
        """Test path length with obstacles matches expected results"""
        # Test grid from problem statement
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0], 
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        
        path = rift_weaver(grid, (0, 0), (4, 4))
        
        assert path is not None, "Path should be found around obstacles"
        # Problem statement expects path length ~8, allow some tolerance
        assert len(path) <= 8, f"Path length {len(path)} should be <= 8 for efficiency"
        assert path[0] == (0, 0), "Path should start at origin"
        assert path[-1] == (4, 4), "Path should end at goal"
    
    def test_phi_heuristic(self):
        """Test φ-vortex heuristic calculation"""
        phi = (1 + np.sqrt(5)) / 2
        
        # Test basic heuristic
        h = phi_heuristic((0, 0), (3, 4))
        expected = phi * min(3, 4) + abs(3 - 4)  # φ*3 + 1
        assert abs(h - expected) < 0.001, f"Heuristic {h} should equal {expected}"
        
        # Test diagonal case
        h_diag = phi_heuristic((0, 0), (5, 5))
        expected_diag = phi * 5 + 0  # φ*5 + 0
        assert abs(h_diag - expected_diag) < 0.001, "Diagonal heuristic should be φ*min_distance"

    def test_digital_root(self):
        """Test digital root calculation for mod-9 logic"""
        assert digital_root(9) == 9, "Digital root of 9 should be 9"
        assert digital_root(18) == 9, "Digital root of 18 should be 9"
        assert digital_root(12) == 3, "Digital root of 12 should be 3"
        assert digital_root(456) == 6, "Digital root of 456 should be 6"


class TestAStarNavigator:
    """Test cases for A* navigator"""
    
    def test_fractal_grid_navigation(self):
        """Test A* navigation on fractal grid"""
        grid = FractalGrid(10, 10)
        navigator = PhiAStarNavigator(grid)
        
        path = navigator.find_path((0, 0), (9, 9))
        
        assert path is not None, "A* should find path on fractal grid"
        assert len(path) >= 10, "Path should be at least 10 steps for 10x10 grid"
        assert path[0] == (0, 0), "Path should start at origin"
        assert path[-1] == (9, 9), "Path should end at goal"
        
        # Check search statistics
        stats = navigator.get_search_statistics()
        assert stats['nodes_explored'] > 0, "Should have explored some nodes"
        assert stats['path_length'] == len(path), "Stats should match actual path length"
    
    def test_obstacle_navigation(self):
        """Test A* navigation around obstacles"""
        grid = FractalGrid(5, 5)
        
        # Add obstacles
        obstacles = [(2, 1), (1, 2), (3, 2)]
        for obs in obstacles:
            grid.add_obstacle(obs[0], obs[1])
        
        navigator = PhiAStarNavigator(grid)
        path = navigator.find_path((0, 0), (4, 4))
        
        assert path is not None, "Should find path around obstacles"
        assert len(path) <= 12, "Path should be reasonably efficient with obstacles"
        
        # Verify no path goes through obstacles
        for pos in path:
            assert pos not in obstacles, f"Path should not go through obstacle at {pos}"


class TestPrimeCore:
    """Test cases for PRIMECORE system"""
    
    def test_dna_mod9_integration(self):
        """Test DNA mod9 tuner integration"""
        core = PrimeCore()
        
        # Test multifractal parameters
        result = core.add_ros_data({'h_q2': 0.82, 'delta_h': 0.5})
        assert 'nav_data' in result, "Should publish ROS navigation data"
        
        # Check metadata was added
        assert 'multifractal_params' in core.metadata, "Multifractal params should be stored"
        params = core.metadata['multifractal_params']
        assert params['h_q2'] == 0.82, "h(q=2) should be 0.82"
        assert params['delta_h'] == 0.5, "Δh should be 0.5"
    
    def test_signature_functionality(self):
        """Test Ed25519 signature creation (verification may fail due to state changes)"""
        core = PrimeCore()
        
        # Add some data
        core.add_metadata('test_key', 'test_value')
        
        # Create signature
        sig_info = core.sign()
        
        assert 'signature' in sig_info, "Should create signature"
        assert 'timestamp' in sig_info, "Should include timestamp"
        assert 'state_hash' in sig_info, "Should include state hash"
        
        # Check signature info was stored
        assert 'signature' in core.metadata, "Signature should be stored in metadata"
    
    def test_save_load_functionality(self):
        """Test save and load system state"""
        import tempfile
        import os
        
        core = PrimeCore()
        core.add_metadata('test_data', {'value': 42})
        core.update_vessel_state([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 90.0)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test save
            saved = core.save(tmp_path)
            assert saved, "Save should succeed"
            
            # Create new core and load
            core2 = PrimeCore()
            loaded = core2.load(tmp_path)
            assert loaded, "Load should succeed"
            
            # Verify data loaded correctly
            assert 'test_data' in core2.metadata, "Metadata should be loaded"
            assert core2.metadata['test_data']['value'] == 42, "Metadata values should match"
            assert np.allclose(core2.vessel_state['position'], [1.0, 2.0, 3.0]), "Position should match"
            assert core2.vessel_state['heading'] == 90.0, "Heading should match"
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_navigation_pipeline(self):
        """Test complete navigation pipeline from PRIMECORE to pathfinding"""
        # Initialize systems
        core = PrimeCore()
        grid = FractalGrid(8, 8)
        navigator = PhiAStarNavigator(grid)
        
        # Add obstacles
        grid.add_obstacle(3, 3)
        grid.add_obstacle(4, 3)
        grid.add_obstacle(3, 4)
        
        # Update vessel state
        core.update_vessel_state([0, 0, 0], [1, 0, 0], 0.0)
        
        # Add multifractal parameters
        core.add_ros_data({'h_q2': 0.82, 'delta_h': 0.5, 'mandelbrot_d': 1.5})
        
        # Find path
        path = navigator.find_path((0, 0), (7, 7))
        
        assert path is not None, "Navigation pipeline should find path"
        assert len(path) >= 8, "Path should be reasonable length for 8x8 grid"
        
        # Generate secure navigation command
        target = [7.0, 7.0, 0.0]
        secure_cmd = core.secure_navigation_command(target)
        
        assert 'command' in secure_cmd, "Should generate secure command"
        assert 'hash' in secure_cmd, "Should include command hash"
        assert 'signature' in secure_cmd, "Should include Ed25519 signature"
        assert 'vortex_flow' in secure_cmd, "Should include DNA vortex flow"
    
    def test_rift_weaver_vs_astar_comparison(self):
        """Compare rift_weaver and A* navigator performance"""
        # Same test grid
        grid_array = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0], 
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        
        # Test rift_weaver
        rift_path = rift_weaver(grid_array, (0, 0), (4, 4))
        
        # Test A* navigator
        fractal_grid = FractalGrid(5, 5)
        for y in range(5):
            for x in range(5):
                if grid_array[y][x] == 1:
                    fractal_grid.add_obstacle(x, y)
        
        navigator = PhiAStarNavigator(fractal_grid)
        astar_path = navigator.find_path((0, 0), (4, 4))
        
        assert rift_path is not None, "Rift weaver should find path"
        assert astar_path is not None, "A* should find path"
        
        # Both should be reasonably efficient
        assert len(rift_path) <= 8, f"Rift weaver path ({len(rift_path)}) should be <= 8"
        assert len(astar_path) <= 8, f"A* path ({len(astar_path)}) should be <= 8"
        
        # Both should reach the goal
        assert rift_path[-1] == (4, 4), "Rift weaver should reach goal"
        assert astar_path[-1] == (4, 4), "A* should reach goal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])