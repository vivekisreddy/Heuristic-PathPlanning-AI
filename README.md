# AI Path Planning & Heuristic Optimization

![cover](banner.svg)

A Python/C++ implementation of classical AI search algorithms for robot navigation in 2D grid environments, with a focus on comparing optimal and local search strategies under obstacle constraints.

---

## Overview

This project implements and analyzes two families of heuristic search:

- **A\*** — optimal pathfinding using a Manhattan distance heuristic, guaranteeing the shortest path from start to goal
- **Hill Climbing variants** — local search strategies that greedily optimize toward the goal, with analysis of convergence behavior and failure modes (local minima, plateaus, ridges) in obstacle-heavy environments

The goal was not just to run the algorithms, but to study *how* and *when* they break — particularly how obstacle density and placement affect convergence, optimality, and completeness.

---

## Algorithms

### A* Search
- Evaluation function: `f(n) = g(n) + h(n)` — actual cost from start + Manhattan distance heuristic
- Explores the lowest-cost frontier node at each step
- Guarantees optimal path when heuristic is admissible
- Tracks explored set to avoid revisiting nodes

### Hill Climbing Variants
- **Steepest-ascent** — evaluates all neighbors, moves to the best
- **Stochastic** — randomly selects among uphill moves
- **Random-restart** — reruns from random start positions to escape local minima
- Convergence behavior logged and compared across obstacle configurations

---

## Environment

- Configurable 2D grid (NxN)
- Randomized or manually placed obstacle layouts
- Start and goal positions configurable via input
- Visualization of explored nodes, frontier, and final path

---

## Results

| Algorithm | Optimal? | Complete? | Notes |
|---|---|---|---|
| A* | Yes | Yes | Consistent across all obstacle configs |
| Steepest Hill Climb | No | No | Fails at local minima |
| Stochastic Hill Climb | No | No | Less deterministic failure |
| Random-restart Hill Climb | No | Probabilistic | Recovery improves with restarts |

---

## Technologies

- **Python** — algorithm implementations, environment simulation, visualization
- **C/C++** — performance-critical grid operations
- **Scikit-Learn** — supporting data analysis and experiment logging
- **Matplotlib** — path and convergence visualization

---

## Repository Structure

```
/algorithms        # A*, hill climbing implementations
/environments      # Grid environment and obstacle generation
/experiments       # Convergence analysis scripts
/utils             # Heuristics, logging, visualization helpers
```

---

## Usage

```bash
git clone https://github.com/vivekisreddy/AI-Path-Planning.git
cd AI-Path-Planning

# Run A* on a default grid
python algorithms/astar.py --grid 20 --obstacles 0.3

# Run hill climbing comparison
python algorithms/hill_climb.py --variant steepest --restarts 10
```

---

## Author

**Vivek Reddy Kasireddy** — [LinkedIn](https://linkedin.com) | [Website](https://vivek.com)  
WPI Computer Science & Robotics Engineering, Class of 2026
