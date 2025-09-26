# AurobotNav

Aurobot Blueprint: PRIMECORE secure vessels + φ-A* nav + DNA vortex math for catastrophe evasion

## Overview

This repository implements the **Trance Vortex Equation** for advanced pathfinding based on psytrance-inspired mathematical models. The system combines golden ratio (φ) enhanced A* algorithms with multifractal scaling for optimal navigation in complex environments.

## Key Components

### RiftWeaver Algorithm (`rift_weaver.py`)
- **φ-Enhanced A* Pathfinding**: Uses golden ratio (φ = 1.618...) for superior heuristic calculations
- **Multifractal Amplification**: Applies h(q=2) = 0.82 scaling factor for DNA-like path optimization  
- **Vortex Harmony**: Digital root mod9 flux detection for 3-6-9 harmonic bonuses
- **Trance Entropy Integration**: Based on equation E ≈ 3.807 for phantom resonance detection

### ROS2 Integration (`auro_ros_node.py`)
- **PRIMECORE Navigation Node**: Publishes φ-enhanced paths to `/auro_path` topic
- **Real-time Path Generation**: Integrates RiftWeaver algorithm for dynamic pathfinding
- **Fallback Compatibility**: Works with or without ROS2 installation
- **Trance-Entropy Ready**: Compatible with 30-day phantom persistence systems

## Mathematical Foundation

The implementation is based on the Trance Vortex Equation:
```
E = mean_ratio * e^(h(q=2) * ln(φ^{-1})) * (DR mod9) ≈ 3.807
```

Where:
- φ = 1.618034 (golden ratio)  
- h(q=2) = 0.82 (multifractal scaling)
- DR mod9 = digital root modulo 9 for flux detection
- Multifractal amplification ≈ 1.484

## Usage

### Basic Pathfinding
```python
from rift_weaver import rift_weaver

grid = [[0,1,0,0,0], [0,0,1,0,1], [1,0,0,1,0], [0,1,0,0,0], [0,0,1,0,0]]
path = rift_weaver(grid, (0,0), (4,4))
print(f"φ-enhanced path: {path}")
```

### ROS2 Node
```bash
python auro_ros_node.py  # Publishes to /auro_path topic
```

## Testing

Run comprehensive tests:
```bash
python rift_weaver.py    # Tests multiple pathfinding scenarios
python auro_ros_node.py  # Tests ROS2 integration (with fallback)
```

## License

MIT License - Open source implementation of wave genetics and trance entropy mathematics.
