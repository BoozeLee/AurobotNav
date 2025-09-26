"""
RiftWeaver: φ-Enhanced A* Pathfinding Algorithm
Based on the Trance Vortex Equation for multifractal navigation

This implementation uses:
- φ (phi) = 1.618... (golden ratio) for enhanced heuristics
- Digital root mod9 flux for vortex harmony detection
- Multifractal amplification h(q=2) = 0.82 for DNA-like path scaling
- Trance entropy modeling for PRIMECORE navigation systems
"""

import heapq
import math

# Constants from the Trance Vortex Equation
phi = (1 + math.sqrt(5)) / 2  # 1.618 - Golden ratio
h_q2 = 0.82  # Multifractal scaling factor from DNA simulations

def digital_root(n):
    """Calculate digital root for mod9 flux detection"""
    return 1 + (n - 1) % 9

def phi_heuristic(a, b, phi=phi):
    """φ-enhanced Manhattan distance heuristic"""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return phi * min(dx, dy) + abs(dx - dy)

def rift_weaver(grid, start, goal):
    """
    φ-Enhanced A* pathfinding with multifractal amplification
    
    Args:
        grid: 2D array where 0=passable, 1=obstacle
        start: (x, y) starting position  
        goal: (x, y) goal position
        
    Returns:
        List of (x, y) coordinates representing the optimal φ-path
    """
    open_set = [(0, start)]
    g_score = {start: 0}
    f_score = {start: phi_heuristic(start, goal)}
    came_from = {}

    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            # Check bounds and passability
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                # Calculate vortex harmony based on digital root
                dr_step = digital_root(abs(dx) + abs(dy))
                skip_bonus = 0.8 if dr_step in [3,6,9] else 1.0  # Vortex harmony bonus for 3-6-9 flux
                
                # Base movement cost (1 for orthogonal, √2 for diagonal)
                base_cost = 1 if abs(dx) + abs(dy) == 1 else math.sqrt(2)
                
                # Apply multifractal amplification: e^(h(q=2) * ln(φ)) ≈ 1.33
                multifractal_amp = math.exp(h_q2 * math.log(phi))
                
                cost = base_cost * skip_bonus * multifractal_amp
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + phi_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def test_rift_weaver():
    """Test the RiftWeaver algorithm with multiple scenarios"""
    print("Testing RiftWeaver φ-enhanced A* algorithm...")
    
    # Test 1: 5x5 fractal grid from problem statement
    print("\nTest 1: 5x5 fractal grid")
    grid = [[0,1,0,0,0], [0,0,1,0,1], [1,0,0,1,0], [0,1,0,0,0], [0,0,1,0,0]]
    path = rift_weaver(grid, (0,0), (4,4))
    print(f"Path from (0,0) to (4,4): {path}")
    
    # Test 2: Simple 3x3 grid
    print("\nTest 2: 3x3 open grid")
    simple_grid = [[0,0,0], [0,0,0], [0,0,0]]
    path2 = rift_weaver(simple_grid, (0,0), (2,2))
    print(f"Path from (0,0) to (2,2): {path2}")
    
    # Test 3: No path scenario
    print("\nTest 3: Blocked path")
    blocked_grid = [[0,1,0], [1,1,1], [0,1,0]]
    path3 = rift_weaver(blocked_grid, (0,0), (2,2))
    print(f"Path from (0,0) to (2,2) when blocked: {path3}")
    
    # Calculate some metrics
    print(f"\nφ (phi) constant: {phi:.6f}")
    print(f"h(q=2) multifractal factor: {h_q2}")
    print(f"Multifractal amplification: {math.exp(h_q2 * math.log(phi)):.3f}")

if __name__ == "__main__":
    test_rift_weaver()