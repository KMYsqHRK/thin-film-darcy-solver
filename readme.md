# Standalone Pressure Distribution Solver

This is a standalone version of the thin-film pressure distribution solver that does not require the ParticleWorks SDK.

## Overview

The solver takes time-series edge pressure data as input and computes:
- Pressure distribution across a 2D grid using Poisson equation solver
- Lift forces (sum of vertical forces from pressure)
- Torques about a specified rotation center
- Optional pressure grid output at specified intervals

## Key Features

- **Input**: CSV file with time-series edge pressure data
- **Output**:
  - Lift forces and torques over time (CSV)
  - Pressure distribution grid (optional, CSV)
- **Solver**: Iterative BiCGSTAB with IncompleteLUT preconditioner
- **Physics**: Darcy's law infiltration modeling with wet/dry dynamics

## Files

- `example_coupling_pressuredist_friction.hpp` - Header file with Solver class
- `example_coupling_pressuredist_friction.cpp` - Implementation
- `standalone_example_main.cpp` - Example main program
- `example_settings.json` - Example configuration file
- `example_input_edge_pressure.csv` - Example input data format

## Building

### Prerequisites
- C++11 or later compiler
- Eigen library (for linear algebra)
- nlohmann/json library (included in third_party/)

### Compile
```bash
g++ -std=c++11 -I. -I/path/to/eigen standalone_example_main.cpp example_coupling_pressuredist_friction.cpp -o pressure_solver
```

Or using the provided Makefile:
```bash
make standalone
```

## Usage

### Basic Usage
```bash
./pressure_solver [settings_file] [input_file]
```

### Example
```bash
./pressure_solver example_settings.json example_input_edge_pressure.csv
```

## Input Format

### Settings File (JSON)

```json
{
    "input_pressure_file": "input_edge_pressure.csv",
    "output_lift_force_file": "output_lift_forces.csv",
    "output_pressure_grid_file": "output_pressure_grid.csv",

    "division_number": 10,

    "boundary_points": [
        {"name": "Point1", "position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        {"name": "Point2", "position": {"x": 1.0, "y": 0.0, "z": 0.0}},
        {"name": "Point3", "position": {"x": 0.0, "y": 1.0, "z": 0.0}}
    ],

    "physics": {"mass": 10.0},
    "gravity": {"x": 0.0, "y": 0.0, "z": 9.8},
    "rotation_center": {"x": 0.5, "y": 0.5, "z": 0.0},

    "infiltration": {
        "v_infil_coefficient": 1.0,
        "hydraulic_conductivity": 0.01,
        "kinematic_viscosity_coefficient": 1.0
    },

    "pressure_grid_export": {
        "save_interval": 10,
        "include_wet_dry": true
    },

    "solver": {
        "max_iterations": 1000,
        "tolerance": 1e-6
    }
}
```

#### Key Parameters:

- `division_number`: Grid resolution (n×n interior points, (n+1)×(n+1) total)
- `boundary_points`: Three points defining the parallelogram domain
  - Point 1: Origin
  - Point 2: Defines X-direction edge
  - Point 3: Defines Y-direction edge
- `mass`: Object mass for force calculations (kg)
- `rotation_center`: Center point for torque calculations
- `hydraulic_conductivity`: Darcy's law permeability coefficient
- `solver.max_iterations`: Maximum iterations for iterative solver
- `solver.tolerance`: Convergence tolerance for iterative solver

### Input Edge Pressure File (CSV)

Format: `Time,P1,P2,P3,...,P(4n)`

- First column: Time value
- Remaining 4n columns: Pressure values at boundary points
  - n pressures along bottom edge (left to right)
  - n pressures along right edge (bottom to top)
  - n pressures along top edge (right to left)
  - n pressures along left edge (top to bottom)

Example for `division_number=10` (requires 40 pressure values):
```csv
Time,P1,P2,...,P40
0.00,100,100,...,100
0.01,105,105,...,105
0.02,110,110,...,110
```

## Output Format

### Lift Force Output (CSV)

Format: `Time,LiftForce,Torque`

```csv
Time,LiftForce,Torque
0.00,245.0,122.5
0.01,257.25,128.625
0.02,269.5,134.75
```

- `Time`: Simulation time
- `LiftForce`: Total vertical lift force (N)
- `Torque`: Torque about rotation center (N·m)

### Pressure Grid Output (CSV)

Format: `Time,i,j,x,y,z,Pressure[,WetDry]`

```csv
Time,i,j,x,y,z,Pressure,WetDry
0.00,0,0,0.0,0.0,0.0,100.0,1.0
0.00,0,1,0.1,0.0,0.0,105.2,1.0
...
```

- Saved at intervals specified by `pressure_grid_save_interval`
- `(i,j)`: Grid indices
- `(x,y,z)`: Physical coordinates
- `Pressure`: Computed pressure value (Pa)
- `WetDry`: 1.0 = wet, 0.0 = dry (optional, if `include_wet_dry` is true)

## Algorithm

1. **Initialization**: Set up grid, distance matrices, and wet/dry states
2. **For each timestep**:
   a. Create boundary pressure matrix from input data
   b. Solve Poisson equation for interior pressure distribution
   c. Update infiltration distance using Darcy's law
   d. Update wet/dry states based on infiltration spreading
   e. Calculate lift force and torque from pressure distribution
   f. Save outputs

### Poisson Equation Solver

- Uses compressed system with only wet points
- Applies P=0 Dirichlet boundary condition at wet-dry interfaces
- BiCGSTAB iterative solver with IncompleteLUT preconditioner
- 9-point stencil for improved accuracy

### Wet/Dry Dynamics

- Models infiltration spreading similar to Conway's Game of Life
- Cells become wet when infiltration distance exceeds threshold
- Spreading to neighbors (4-connected and 8-connected) based on infiltration distance
- Dry cells treated as atmospheric pressure (P=0)

## Code Structure

```
standalone::pressure_solver::
├── Solver (main class)
│   ├── load_settings()         - Load JSON configuration
│   ├── load_input_data()       - Load CSV edge pressure data
│   ├── process_all_timesteps() - Process all timesteps
│   ├── save_lift_force_output()- Save lift force results
│   │
│   ├── process_single_timestep() - Process one timestep
│   ├── solve_pressure_distribution() - Solve Poisson equation
│   ├── calculate_lift_force_and_torque() - Compute forces
│   ├── update_wet_dry_matrix() - Update wet/dry states
│   └── update_infil_distance_matrix() - Update infiltration
│
├── TimestepData (struct)       - Input data for one timestep
├── Settings (struct)           - Configuration parameters
└── BoundaryPoint (struct)      - Boundary point definition
```

## Differences from SDK Version

The standalone version removes:
- ParticleWorks SDK dependencies
- Particle-based pressure extraction (`calculate_pressure_at_point`)
- Equations of motion solver (`calculate_friction`, `set_node_position`)
- Dynamic object motion tracking

The standalone version adds:
- CSV input/output for edge pressures
- Simplified interface focused on pressure-to-force conversion
- Self-contained execution without external solver

## Performance Notes

- Grid size: O((n+1)² ) points
- Solver complexity: O(iterations × non-zeros)
- Typical convergence: 10-100 iterations for 5×5 to 50×50 grids
- Memory: ~O(n²) for matrices

Recommended grid sizes:
- Small: n=5-10 (fast, lower accuracy)
- Medium: n=10-20 (balanced)
- Large: n=20-50 (accurate, slower)

## License

See main project LICENSE file.

## Contact

For questions or issues, please refer to the main project repository.
