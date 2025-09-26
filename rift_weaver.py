"""
rift_weaver.py - Enhanced A* 8-way for multifractal rifts
h_φ = φ*min(|dx|,|dy|) + |dx-dy| (φ=(1+√5)/2). Mod-9 DR skips 3-6-9 cost=0. 
g amp e^(0.82*ln(φ))≈1.33. Mandelbrot D=1.5 cost for edges.
"""

import heapq
import math
import numpy as np
from collections import defaultdict
import time

# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618

# Multifractal constants
H_Q2 = 0.82  # Hölder exponent h(q=2) = 0.82
MANDELBROT_D = 1.5  # Mandelbrot dimension D = 1.5

def digital_root(n):
    """Calculate digital root (DR) for mod-9 calculations"""
    if n == 0:
        return 0
    return 1 + (n - 1) % 9

def phi_heuristic(current, goal):
    """φ-optimized heuristic: h_φ = φ*min(|dx|,|dy|) + |dx-dy|"""
    dx = abs(current[0] - goal[0])
    dy = abs(current[1] - goal[1])
    return PHI * min(dx, dy) + abs(dx - dy)

def rift_weaver(grid, start, goal):
    """
    Enhanced A* pathfinder with multifractal rift navigation
    8-way movement with φ-heuristic and mod-9 digital root optimizations
    """
    if not grid or not grid[0]:
        return []
    
    rows, cols = len(grid), len(grid[0])
    
    # Validate start and goal positions
    if (not (0 <= start[0] < rows and 0 <= start[1] < cols) or
        not (0 <= goal[0] < rows and 0 <= goal[1] < cols) or
        grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1):
        return []
    
    # A* data structures
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    f_score = defaultdict(lambda: float('inf'))
    f_score[start] = phi_heuristic(start, goal)
    
    # 8-directional movement
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    nodes_explored = 0
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        nodes_explored += 1
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check bounds and obstacles
            if (not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols) or
                grid[neighbor[0]][neighbor[1]] == 1):
                continue
            
            # Calculate movement cost with multifractal enhancements
            base_cost = math.sqrt(dx*dx + dy*dy)  # Euclidean distance
            
            # Digital root calculation for mod-9 optimization
            dr_step = digital_root(nodes_explored)
            
            # Skip cost for mod-9 DR values 3, 6, 9 (cost = 0)
            if dr_step in [3, 6, 9]:
                skip_bonus = 0.0
            else:
                skip_bonus = base_cost
            
            # Apply multifractal amplification: g_amp = e^(h(q=2)*ln(φ)) ≈ 1.33
            exp_factor = math.exp(H_Q2 * math.log(PHI))  # ≈ 1.33
            
            # Mandelbrot D=1.5 cost adjustment for edges
            mandelbrot_cost = base_cost ** MANDELBROT_D
            
            # Final cost calculation
            total_cost = (skip_bonus + mandelbrot_cost) * exp_factor
            
            tentative_g_score = g_score[current] + total_cost
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + phi_heuristic(neighbor, goal)
                
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # No path found

# Test implementation
if __name__ == "__main__":
    print("=== RiftWeaver Enhanced A* Test ===")
    
    # Create 5x5 test grid
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start = (0, 0)
    goal = (4, 4)
    
    print(f"Grid:")
    for row in grid:
        print(row)
    print(f"\nFinding path from {start} to {goal}")
    
    # Run pathfinding
    start_time = time.time()
    path = rift_weaver(grid, start, goal)
    end_time = time.time()
    
    if path:
        print(f"Path found: {path}")
        print(f"Path length: {len(path)}")
        print(f"Expected: 8 nodes, 18 steps (from problem statement)")
        print(f"Computation time: {(end_time - start_time)*1000:.2f}ms")
        
        # Test phi heuristic
        test_h = phi_heuristic(start, goal)
        print(f"φ-heuristic h_φ({start} -> {goal}) = {test_h:.3f}")
        
        # Test digital root
        for i in range(1, 10):
            dr = digital_root(i)
            print(f"DR({i}) = {dr}", end=" ")
        print()
        
        # Calculate actual path cost
        total_cost = 0
        for i in range(len(path) - 1):
            current, next_pos = path[i], path[i+1]
            dx = next_pos[0] - current[0]
            dy = next_pos[1] - current[1]
            base_cost = math.sqrt(dx*dx + dy*dy)
            
            # Apply same cost calculation as in algorithm
            dr_step = digital_root(i + 1)
            if dr_step in [3, 6, 9]:
                skip_bonus = 0.0
            else:
                skip_bonus = base_cost
            
            exp_factor = math.exp(H_Q2 * math.log(PHI))
            mandelbrot_cost = base_cost ** MANDELBROT_D
            step_cost = (skip_bonus + mandelbrot_cost) * exp_factor
            total_cost += step_cost
        
        print(f"Total path cost: {total_cost:.3f}")
        
    else:
        print("No path found!")
    
    print("=== RiftWeaver Test Complete ===")