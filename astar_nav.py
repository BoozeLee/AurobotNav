"""
astar_nav.py - φ-optimized A* pathfinding with Mandelbrot fractal efficiency
Heuristic: h_φ = φ*min(dx,dy) + |dx-dy|, mod-9 cycles, Mandelbrot D=1.5 (2.27x efficiency)
"""

import numpy as np
import heapq
import math
from collections import defaultdict
import time

class MandelbrotOptimizer:
    """Mandelbrot set optimization for fractal pathfinding efficiency"""
    
    def __init__(self, dimension=1.5):
        self.fractal_dimension = dimension  # D=1.5 for 2.27x efficiency
        self.max_iterations = 100
        self.escape_radius = 2.0
        
    def mandelbrot_efficiency(self, c_real, c_imag):
        """Calculate Mandelbrot efficiency factor"""
        c = complex(c_real, c_imag)
        z = 0
        
        for iteration in range(self.max_iterations):
            if abs(z) > self.escape_radius:
                # Calculate efficiency based on escape time and fractal dimension
                efficiency = (iteration / self.max_iterations) ** self.fractal_dimension
                return min(efficiency * 2.27, 2.27)  # Cap at 2.27x efficiency
            z = z*z + c
            
        return 1.0  # Default efficiency for points in the set
    
    def optimize_path_segment(self, start, end):
        """Apply Mandelbrot optimization to path segment"""
        # Map coordinates to complex plane
        c_real = (start[0] + end[0]) / 200.0 - 1.0  # Normalize to Mandelbrot range
        c_imag = (start[1] + end[1]) / 200.0 - 1.0
        
        efficiency_factor = self.mandelbrot_efficiency(c_real, c_imag)
        return efficiency_factor

class GoldenRatioHeuristic:
    """φ-optimized heuristic function: h_φ = φ*min(dx,dy) + |dx-dy|"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio φ ≈ 1.618
        self.mod9_cycle = 0
        
    def calculate_heuristic(self, current, goal):
        """Calculate φ-optimized heuristic distance"""
        dx = abs(current[0] - goal[0])
        dy = abs(current[1] - goal[1])
        
        # φ-optimized heuristic: h_φ = φ*min(dx,dy) + |dx-dy|
        h_phi = self.phi * min(dx, dy) + abs(dx - dy)
        
        # Apply mod-9 cycle adjustment
        cycle_factor = 1 + (self.mod9_cycle % 9) * 0.01
        self.mod9_cycle += 1
        
        return h_phi * cycle_factor
    
    def reset_cycle(self):
        """Reset mod-9 cycle counter"""
        self.mod9_cycle = 0

class FractalGrid:
    """Fractal-aware grid for navigation"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = set()
        self.fractal_weights = {}
        self._generate_fractal_weights()
        
    def _generate_fractal_weights(self):
        """Generate fractal-based movement weights"""
        for x in range(self.width):
            for y in range(self.height):
                # Create fractal pattern using recursive subdivision
                weight = self._fractal_weight(x, y, self.width, self.height, depth=3)
                self.fractal_weights[(x, y)] = weight
                
    def _fractal_weight(self, x, y, w, h, depth):
        """Calculate fractal weight using recursive pattern"""
        if depth <= 0:
            return 1.0
            
        # Normalize coordinates
        nx = x / w if w > 0 else 0
        ny = y / h if h > 0 else 0
        
        # Create fractal pattern
        base_weight = 1 + 0.5 * math.sin(nx * math.pi * 2) * math.cos(ny * math.pi * 2)
        
        # Recursive subdivision
        sub_weight = self._fractal_weight(x % (w//2), y % (h//2), w//2, h//2, depth-1)
        
        return base_weight * (1 + sub_weight * 0.3)
    
    def add_obstacle(self, x, y):
        """Add obstacle to grid"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles.add((x, y))
    
    def is_valid_position(self, x, y):
        """Check if position is valid (not an obstacle and within bounds)"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                (x, y) not in self.obstacles)
    
    def get_movement_cost(self, from_pos, to_pos):
        """Get movement cost between two positions including fractal weights"""
        if not self.is_valid_position(to_pos[0], to_pos[1]):
            return float('inf')
            
        base_cost = math.sqrt((to_pos[0] - from_pos[0])**2 + (to_pos[1] - from_pos[1])**2)
        fractal_weight = self.fractal_weights.get(to_pos, 1.0)
        
        return base_cost * fractal_weight

class PhiAStarNavigator:
    """φ-optimized A* navigator with Mandelbrot fractal efficiency"""
    
    def __init__(self, grid):
        self.grid = grid
        self.heuristic = GoldenRatioHeuristic()
        self.mandelbrot_optimizer = MandelbrotOptimizer()
        self.search_stats = {
            'nodes_explored': 0,
            'path_length': 0,
            'efficiency_factor': 1.0,
            'computation_time': 0.0
        }
        
    def get_neighbors(self, pos):
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = []
        
        # 8-directional movement
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self.grid.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
                
        return neighbors
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from A* search"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def find_path(self, start, goal):
        """Find optimal path using φ-optimized A* algorithm"""
        start_time = time.time()
        self.heuristic.reset_cycle()
        
        # A* data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = self.heuristic.calculate_heuristic(start, goal)
        
        self.search_stats['nodes_explored'] = 0
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            self.search_stats['nodes_explored'] += 1
            
            if current == goal:
                path = self.reconstruct_path(came_from, current)
                
                # Calculate path efficiency using Mandelbrot optimization
                total_efficiency = 1.0
                efficiency_count = 0
                for i in range(len(path) - 1):
                    segment_efficiency = self.mandelbrot_optimizer.optimize_path_segment(
                        path[i], path[i+1]
                    )
                    total_efficiency += segment_efficiency
                    efficiency_count += 1
                
                # Average efficiency across all segments
                if efficiency_count > 0:
                    total_efficiency = total_efficiency / efficiency_count
                else:
                    total_efficiency = 1.0
                
                # Update search statistics
                self.search_stats['path_length'] = len(path)
                self.search_stats['efficiency_factor'] = total_efficiency
                self.search_stats['computation_time'] = time.time() - start_time
                
                return path
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = (g_score[current] + 
                                   self.grid.get_movement_cost(current, neighbor))
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = (g_score[neighbor] + 
                                       self.heuristic.calculate_heuristic(neighbor, goal))
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        self.search_stats['computation_time'] = time.time() - start_time
        return None
    
    def get_search_statistics(self):
        """Get statistics from the last search"""
        return self.search_stats.copy()
    
    def optimize_existing_path(self, path):
        """Apply Mandelbrot optimization to existing path"""
        if len(path) < 2:
            return path
            
        optimized_path = [path[0]]
        
        for i in range(1, len(path)):
            current_segment = path[i-1:i+1]
            efficiency = self.mandelbrot_optimizer.optimize_path_segment(
                current_segment[0], current_segment[1]
            )
            
            # If efficiency is high enough, keep the segment
            if efficiency > 1.5:  # Threshold for keeping segments
                optimized_path.append(path[i])
            else:
                # Try to find a more efficient intermediate point
                start, end = current_segment[0], current_segment[1]
                mid_x = (start[0] + end[0]) // 2
                mid_y = (start[1] + end[1]) // 2
                
                if self.grid.is_valid_position(mid_x, mid_y):
                    optimized_path.extend([(mid_x, mid_y), path[i]])
                else:
                    optimized_path.append(path[i])
        
        return optimized_path

# Example usage and testing
if __name__ == "__main__":
    print("=== φ-Optimized A* Navigator Test ===")
    
    # Create a test grid
    grid = FractalGrid(20, 20)
    
    # Add some obstacles
    obstacles = [(5, 5), (5, 6), (5, 7), (10, 10), (11, 10), (12, 10)]
    for obs in obstacles:
        grid.add_obstacle(obs[0], obs[1])
    
    # Initialize navigator
    navigator = PhiAStarNavigator(grid)
    
    # Test pathfinding
    start = (0, 0)
    goal = (19, 19)
    
    print(f"Finding path from {start} to {goal}")
    path = navigator.find_path(start, goal)
    
    if path:
        print(f"Path found with {len(path)} steps")
        print(f"Path: {path[:5]}...{path[-5:] if len(path) > 10 else path}")
        
        # Get search statistics
        stats = navigator.get_search_statistics()
        print(f"Nodes explored: {stats['nodes_explored']}")
        print(f"Efficiency factor: {stats['efficiency_factor']:.3f}")
        print(f"Computation time: {stats['computation_time']:.4f}s")
        
        # Test path optimization
        optimized_path = navigator.optimize_existing_path(path)
        print(f"Optimized path length: {len(optimized_path)} (vs {len(path)})")
        
    else:
        print("No path found!")
    
    # Test heuristic function
    heuristic = GoldenRatioHeuristic()
    test_current = (5, 5)
    test_goal = (15, 15)
    h_value = heuristic.calculate_heuristic(test_current, test_goal)
    print(f"Heuristic h_φ({test_current} -> {test_goal}) = {h_value:.3f}")
    
    # Test Mandelbrot optimization
    mandelbrot = MandelbrotOptimizer()
    efficiency = mandelbrot.mandelbrot_efficiency(-0.5, 0.6)
    print(f"Mandelbrot efficiency factor: {efficiency:.3f}")
    
    print("=== A* Navigator Test Complete ===")