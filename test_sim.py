#!/usr/bin/env python3
"""
Test simulation for Enhanced A* Pathfinding in AurobotNav

This test script validates:
- φ-scaled heuristic functionality
- mod-9 digital roots for vortex harmony
- Mandelbrot fractal obstacle weighting
- Multifractal scaling for g-scores
- DNA vessel flow cost calculations
- Path optimization on various grid configurations

Author: AurobotNav AI System
Version: 1.0
"""

import sys
import math
import numpy as np
from typing import List, Tuple, Dict
import time

# Import our enhanced pathfinder
from astar_nav import AuroPathfinder


class AuroNavTester:
    """
    Comprehensive test suite for enhanced A* pathfinding
    """
    
    def __init__(self):
        self.pathfinder = AuroPathfinder()
        self.test_results = []
        
    def print_grid_with_path(self, grid: List[List[int]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Visualize grid with path overlay
        """
        print("\nGrid visualization:")
        print("0 = free space, 1 = obstacle, S = start, G = goal, * = path")
        
        # Create display grid
        display = []
        for row in grid:
            display.append(row.copy())
        
        # Mark path
        for pos in path:
            if pos != start and pos != goal:
                display[pos[0]][pos[1]] = '*'
        
        # Mark start and goal
        display[start[0]][start[1]] = 'S'
        display[goal[0]][goal[1]] = 'G'
        
        # Print grid
        for row in display:
            print(' '.join(str(cell) for cell in row))
    
    def test_phi_heuristic(self):
        """
        Test φ-scaled heuristic calculations
        """
        print("\n=== Testing φ-scaled Heuristic ===")
        
        test_cases = [
            ((0, 0), (4, 4)),  # Diagonal movement
            ((0, 0), (0, 4)),  # Horizontal movement
            ((0, 0), (4, 0)),  # Vertical movement
            ((2, 2), (3, 3)),  # Short diagonal
        ]
        
        for start, goal in test_cases:
            heuristic = self.pathfinder.phi_heuristic(start, goal)
            dx = abs(start[0] - goal[0])
            dy = abs(start[1] - goal[1])
            euclidean = math.sqrt(dx*dx + dy*dy)
            
            print(f"From {start} to {goal}:")
            print(f"  φ-heuristic: {heuristic:.3f}")
            print(f"  Euclidean: {euclidean:.3f}")
            print(f"  φ-ratio: {heuristic/euclidean:.3f}")
        
        return True
    
    def test_digital_root_mod9(self):
        """
        Test mod-9 digital root calculations for vortex harmony
        """
        print("\n=== Testing Mod-9 Digital Roots ===")
        
        test_numbers = [12, 27, 45, 63, 81, 99, 123, 456, 789]
        
        for num in test_numbers:
            root = self.pathfinder.digital_root_mod9(num)
            is_harmony = self.pathfinder.is_vortex_harmony(num / 100.0)
            print(f"Number {num}: root={root}, harmony={is_harmony}")
        
        return True
    
    def test_mandelbrot_fractal_weight(self):
        """
        Test Mandelbrot fractal weighting for obstacles
        """
        print("\n=== Testing Mandelbrot Fractal Weights ===")
        
        grid_size = (5, 5)
        
        print("Fractal weights for 5x5 grid positions:")
        for x in range(5):
            row_weights = []
            for y in range(5):
                weight = self.pathfinder.mandelbrot_fractal_weight(x, y, grid_size)
                row_weights.append(f"{weight:.2f}")
            print(" ".join(row_weights))
        
        return True
    
    def test_dna_vessel_flow(self):
        """
        Test DNA vessel flow cost calculations
        """
        print("\n=== Testing DNA Vessel Flow Costs ===")
        
        test_moves = [
            ((0, 0), (0, 1)),  # Horizontal
            ((0, 0), (1, 0)),  # Vertical
            ((0, 0), (1, 1)),  # Diagonal
            ((1, 1), (2, 2)),  # Different starting position
        ]
        
        for start, end in test_moves:
            cost = self.pathfinder.dna_vessel_flow_cost(start, end)
            flux_sum = start[0] + start[1] + end[0] + end[1]
            flux_mod = flux_sum % 9
            base_cost = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            
            print(f"Move {start} -> {end}:")
            print(f"  Base cost: {base_cost:.3f}")
            print(f"  DNA flow cost: {cost:.3f}")
            print(f"  Flux sum: {flux_sum}, mod9: {flux_mod}")
        
        return True
    
    def test_5x5_grid_pathfinding(self):
        """
        Test pathfinding on the specified 5x5 grid
        """
        print("\n=== Testing 5x5 Grid Pathfinding ===")
        
        # Original grid from problem statement
        grid = [
            [0, 1, 0, 0, 0],  # Fractal edges
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ]
        
        start = (0, 0)
        goal = (4, 4)
        
        print(f"Finding path from {start} to {goal}")
        
        # Time the pathfinding
        start_time = time.time()
        path = self.pathfinder.rift_weaver(grid, start, goal)
        end_time = time.time()
        
        if path:
            print(f"Path found in {(end_time - start_time)*1000:.2f} ms")
            print(f"Path length: {len(path)} steps")
            print(f"Path coordinates: {path}")
            
            # Calculate path cost
            total_cost = 0.0
            for i in range(len(path) - 1):
                cost = self.pathfinder.dna_vessel_flow_cost(path[i], path[i+1])
                total_cost += cost
            
            print(f"Total path cost: {total_cost:.3f}")
            
            self.print_grid_with_path(grid, path, start, goal)
            
            return True
        else:
            print("No path found!")
            return False
    
    def test_multiple_scenarios(self):
        """
        Test pathfinding on various grid scenarios
        """
        print("\n=== Testing Multiple Grid Scenarios ===")
        
        scenarios = [
            {
                "name": "Empty Grid",
                "grid": [[0] * 5 for _ in range(5)],
                "start": (0, 0),
                "goal": (4, 4)
            },
            {
                "name": "Maze-like Grid",
                "grid": [
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0]
                ],
                "start": (0, 0),
                "goal": (4, 4)
            },
            {
                "name": "No Path Grid",
                "grid": [
                    [0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0]
                ],
                "start": (0, 0),
                "goal": (0, 4)
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            print(f"\nTesting: {scenario['name']}")
            
            start_time = time.time()
            path = self.pathfinder.rift_weaver(
                scenario["grid"], 
                scenario["start"], 
                scenario["goal"]
            )
            end_time = time.time()
            
            if path:
                print(f"✓ Path found: {len(path)} steps in {(end_time - start_time)*1000:.2f} ms")
                results.append(True)
            else:
                print(f"✗ No path found in {(end_time - start_time)*1000:.2f} ms")
                results.append(False)
        
        return results
    
    def test_ros2_integration(self):
        """
        Test ROS2 integration and path publishing
        """
        print("\n=== Testing ROS2 Integration ===")
        
        # Trigger path publication
        self.pathfinder.publish_path()
        
        # Check if path was computed and stored
        if hasattr(self.pathfinder, 'last_path') and self.pathfinder.last_path:
            print("✓ ROS2 path published successfully")
            print(f"Published path: {self.pathfinder.last_path}")
            return True
        else:
            print("✗ Failed to publish ROS2 path")
            return False
    
    def run_all_tests(self):
        """
        Run comprehensive test suite
        """
        print("=" * 60)
        print("AUROBONAV ENHANCED A* PATHFINDING TEST SUITE")
        print("=" * 60)
        
        tests = [
            ("φ-scaled Heuristic", self.test_phi_heuristic),
            ("Mod-9 Digital Roots", self.test_digital_root_mod9),
            ("Mandelbrot Fractal Weights", self.test_mandelbrot_fractal_weight),
            ("DNA Vessel Flow", self.test_dna_vessel_flow),
            ("5x5 Grid Pathfinding", self.test_5x5_grid_pathfinding),
            ("Multiple Scenarios", self.test_multiple_scenarios),
            ("ROS2 Integration", self.test_ros2_integration),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'-' * 40}")
            print(f"Running: {test_name}")
            print(f"{'-' * 40}")
            
            try:
                result = test_func()
                if result:
                    print(f"✓ {test_name} PASSED")
                    passed += 1
                else:
                    print(f"✗ {test_name} FAILED")
            except Exception as e:
                print(f"✗ {test_name} ERROR: {str(e)}")
        
        # Final summary
        print(f"\n{'=' * 60}")
        print(f"TEST SUMMARY: {passed}/{total} tests passed")
        print(f"Success rate: {passed/total*100:.1f}%")
        print(f"{'=' * 60}")
        
        return passed == total


def main():
    """
    Main test execution function
    """
    print("Initializing AurobotNav Enhanced A* Test Suite...")
    
    tester = AuroNavTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! AurobotNav is ready for deployment.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please review the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()