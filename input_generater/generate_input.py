#!/usr/bin/env python3
"""
Input file generator for thin-film Darcy solver.

Generates:
  - settings.json
  - input_edge_pressure.csv  (Dirichlet edges only)

The CSV contains columns only for Dirichlet edges (in edge order:
bottom → right → top → left, skipping Neumann edges).
The C++ solver fills in 0 (zero-flux) for Neumann edges automatically.

Boundary point ordering for division_number = n:
  bottom edge : n points, left to right   (edge index 0)
  right edge  : n points, bottom to top   (edge index 1)
  top edge    : n points, right to left   (edge index 2)
  left edge   : n points, top to bottom   (edge index 3)
"""

import json
import csv
import argparse
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Where this script writes settings.json and the CSV
    # (= the input directory seen by the solver)
    "input_dir": "input",
    "settings_file": "settings.json",
    "pressure_csv_file": "input_edge_pressure.csv",

    # Where the solver writes its output files (created automatically if absent)
    "solver_output_dir": "output",

    # Geometry (three corner points of the parallelogram domain)
    "point1": [-5.0, -5.0, 0.0],   # origin
    "point2": [ 5.0, -5.0, 0.0],   # x-direction edge
    "point3": [-5.0,  5.0, 0.0],   # y-direction edge

    # Grid resolution
    "division_number": 50,

    # Boundary conditions per edge: "dirichlet" or "neumann"
    "bc_bottom": "dirichlet",
    "bc_right":  "neumann",
    "bc_top":    "dirichlet",
    "bc_left":   "neumann",

    # Physics
    "gravity": [0.0, 0.0, 9.8],
    "rotation_center": [0.5, 0.5, 0.0],
    "hydraulic_conductivity": 1.0e-8,
    "viscosity_coefficient": 1.0e-3,

    # Solver
    "max_iterations": 1000,
    "tolerance": 1e-6,

    # Output control
    "pressure_grid_save_interval": 1,
    "pressure_grid_include_wet_dry": True,

    # Time-series parameters
    "t_start": 0.0,
    "t_end":   20.0,
    "dt":      0.002,

    # Pressure profile (applied to Dirichlet edges)
    # "constant" : constant value over time
    # "linear"   : linearly decreasing from p_max to p_min
    # "sine"     : sinusoidal variation
    "pressure_profile": "constant",
    "p_bottom_max": 49000.0,
    "p_bottom_min": 49000.0,
    "p_top_max":    0.0,
    "p_top_min":    0.0,
}


# ---------------------------------------------------------------------------
# Pressure profile helpers
# ---------------------------------------------------------------------------

def make_profile(profile_type: str, p_max: float, p_min: float):
    """Return a function f(t, t_start, t_end) -> pressure value."""
    if profile_type == "constant":
        def f(_t, _t_start, _t_end):
            return p_max
    elif profile_type == "linear":
        def f(t, t_start, t_end):
            if t_end == t_start:
                return p_max
            frac = (t - t_start) / (t_end - t_start)
            return p_max + frac * (p_min - p_max)
    elif profile_type == "sine":
        def f(t, t_start, t_end):
            period = t_end - t_start if t_end != t_start else 1.0
            return p_min + (p_max - p_min) * 0.5 * (1.0 + math.sin(2 * math.pi * (t - t_start) / period))
    else:
        raise ValueError(f"Unknown pressure_profile: '{profile_type}'. "
                         "Choose from: constant, linear, sine")
    return f


# ---------------------------------------------------------------------------
# Settings JSON generation
# ---------------------------------------------------------------------------

def build_settings(p) -> dict:
    return {
        "input_dir":  ".",                 # CSV is in the same dir as settings.json
        "output_dir": p["solver_output_dir"],

        "input_pressure_file":      p["pressure_csv_file"],
        "output_lift_force_file":   "output_lift_forces.csv",
        "output_pressure_grid_file":"output_pressure_grid.csv",

        "division_number": p["division_number"],

        "boundary_points": [
            {"name": "Point1", "position": {"x": p["point1"][0], "y": p["point1"][1], "z": p["point1"][2]}},
            {"name": "Point2", "position": {"x": p["point2"][0], "y": p["point2"][1], "z": p["point2"][2]}},
            {"name": "Point3", "position": {"x": p["point3"][0], "y": p["point3"][1], "z": p["point3"][2]}},
        ],

        "gravity": {"x": p["gravity"][0], "y": p["gravity"][1], "z": p["gravity"][2]},

        "rotation_center": {
            "x": p["rotation_center"][0],
            "y": p["rotation_center"][1],
            "z": p["rotation_center"][2],
        },

        "boundary_conditions": {
            "bottom": {"type": p["bc_bottom"]},
            "right":  {"type": p["bc_right"]},
            "top":    {"type": p["bc_top"]},
            "left":   {"type": p["bc_left"]},
        },

        "infiltration": {
            "hydraulic_conductivity": p["hydraulic_conductivity"],
            "viscosity_coefficient":  p["viscosity_coefficient"],
        },

        "pressure_grid_export": {
            "save_interval":  p["pressure_grid_save_interval"],
            "include_wet_dry": p["pressure_grid_include_wet_dry"],
        },

        "solver": {
            "max_iterations": p["max_iterations"],
            "tolerance":      p["tolerance"],
        },
    }


# ---------------------------------------------------------------------------
# Edge pressure CSV generation
# ---------------------------------------------------------------------------

# Edge name → index in [bottom, right, top, left]
EDGE_ORDER = ["bottom", "right", "top", "left"]

def build_pressure_header_and_rows(p) -> tuple[list[str], list[list]]:
    """
    Return (header, rows) containing only Dirichlet edge columns.
    Neumann edges are omitted; the C++ solver fills them with 0 (zero flux).
    """
    n = p["division_number"]
    t_start = p["t_start"]
    t_end   = p["t_end"]
    dt      = p["dt"]

    bc = {
        "bottom": p["bc_bottom"],
        "right":  p["bc_right"],
        "top":    p["bc_top"],
        "left":   p["bc_left"],
    }

    profiles = {
        "bottom": make_profile(p["pressure_profile"], p["p_bottom_max"], p["p_bottom_min"]),
        "right":  None,  # Neumann by default
        "top":    make_profile(p["pressure_profile"], p["p_top_max"],    p["p_top_min"]),
        "left":   None,  # Neumann by default
    }

    # Build header: only Dirichlet edges
    header = ["Time"]
    col_counter = 1
    dirichlet_edges = []
    for edge in EDGE_ORDER:
        if bc[edge] == "dirichlet":
            for _ in range(n):
                header.append(f"P{col_counter}")
                col_counter += 1
            dirichlet_edges.append(edge)

    # Build rows
    rows = []
    t = t_start
    while t <= t_end + 1e-12:
        t_clamped = min(t, t_end)

        row = [round(t_clamped, 10)]
        for edge in EDGE_ORDER:
            if bc[edge] == "dirichlet":
                val = profiles[edge](t_clamped, t_start, t_end)
                row += [round(val, 4)] * n

        rows.append(row)

        if t >= t_end - 1e-12:
            break
        t = round(t + dt, 10)

    return header, rows


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate settings.json and input_edge_pressure.csv for the thin-film Darcy solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # File paths
    parser.add_argument("--input-dir", default=DEFAULTS["input_dir"],
                        help="Directory where settings.json and CSV are written (= solver input dir)")
    parser.add_argument("--solver-output-dir", default=DEFAULTS["solver_output_dir"],
                        help="Directory where the solver writes its output files "
                             "(written into settings.json; created automatically at runtime)")
    parser.add_argument("--settings-file", default=DEFAULTS["settings_file"],
                        help="Settings JSON filename")
    parser.add_argument("--csv-file",      default=DEFAULTS["pressure_csv_file"],
                        help="Edge pressure CSV filename")

    # Geometry
    parser.add_argument("--division-number", type=int, default=DEFAULTS["division_number"],
                        help="Grid resolution (n×n interior points)")
    parser.add_argument("--point1", nargs=3, type=float, metavar=("X","Y","Z"),
                        default=DEFAULTS["point1"],
                        help="Origin corner of domain")
    parser.add_argument("--point2", nargs=3, type=float, metavar=("X","Y","Z"),
                        default=DEFAULTS["point2"],
                        help="X-direction corner of domain")
    parser.add_argument("--point3", nargs=3, type=float, metavar=("X","Y","Z"),
                        default=DEFAULTS["point3"],
                        help="Y-direction corner of domain")

    # Boundary conditions
    parser.add_argument("--bc-bottom", default=DEFAULTS["bc_bottom"],
                        choices=["dirichlet","neumann"], help="BC type for bottom edge")
    parser.add_argument("--bc-right",  default=DEFAULTS["bc_right"],
                        choices=["dirichlet","neumann"], help="BC type for right edge")
    parser.add_argument("--bc-top",    default=DEFAULTS["bc_top"],
                        choices=["dirichlet","neumann"], help="BC type for top edge")
    parser.add_argument("--bc-left",   default=DEFAULTS["bc_left"],
                        choices=["dirichlet","neumann"], help="BC type for left edge")

    # Physics
    parser.add_argument("--hydraulic-conductivity", type=float,
                        default=DEFAULTS["hydraulic_conductivity"])
    parser.add_argument("--viscosity-coefficient", type=float,
                        default=DEFAULTS["viscosity_coefficient"])
    parser.add_argument("--gravity", nargs=3, type=float, metavar=("X","Y","Z"),
                        default=DEFAULTS["gravity"])
    parser.add_argument("--rotation-center", nargs=3, type=float, metavar=("X","Y","Z"),
                        default=DEFAULTS["rotation_center"])

    # Time series
    parser.add_argument("--t-start", type=float, default=DEFAULTS["t_start"])
    parser.add_argument("--t-end",   type=float, default=DEFAULTS["t_end"])
    parser.add_argument("--dt",      type=float, default=DEFAULTS["dt"])

    # Pressure profile (Dirichlet edges only)
    parser.add_argument("--pressure-profile", default=DEFAULTS["pressure_profile"],
                        choices=["constant","linear","sine"],
                        help="Time variation of boundary pressure")
    parser.add_argument("--p-bottom-max", type=float, default=DEFAULTS["p_bottom_max"],
                        help="Initial (max) pressure on bottom edge")
    parser.add_argument("--p-bottom-min", type=float, default=DEFAULTS["p_bottom_min"],
                        help="Final (min) pressure on bottom edge")
    parser.add_argument("--p-top-max",    type=float, default=DEFAULTS["p_top_max"],
                        help="Initial (max) pressure on top edge")
    parser.add_argument("--p-top-min",    type=float, default=DEFAULTS["p_top_min"],
                        help="Final (min) pressure on top edge")

    # Solver
    parser.add_argument("--max-iterations", type=int, default=DEFAULTS["max_iterations"])
    parser.add_argument("--tolerance",      type=float, default=DEFAULTS["tolerance"])

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    p = {
        "input_dir":                  args.input_dir,
        "solver_output_dir":          args.solver_output_dir,
        "settings_file":              args.settings_file,
        "pressure_csv_file":          args.csv_file,
        "division_number":            args.division_number,
        "point1":                     args.point1,
        "point2":                     args.point2,
        "point3":                     args.point3,
        "bc_bottom":                  args.bc_bottom,
        "bc_right":                   args.bc_right,
        "bc_top":                     args.bc_top,
        "bc_left":                    args.bc_left,
        "gravity":                    args.gravity,
        "rotation_center":            args.rotation_center,
        "hydraulic_conductivity":     args.hydraulic_conductivity,
        "viscosity_coefficient":      args.viscosity_coefficient,
        "max_iterations":             args.max_iterations,
        "tolerance":                  args.tolerance,
        "pressure_grid_save_interval":DEFAULTS["pressure_grid_save_interval"],
        "pressure_grid_include_wet_dry": DEFAULTS["pressure_grid_include_wet_dry"],
        "t_start":                    args.t_start,
        "t_end":                      args.t_end,
        "dt":                         args.dt,
        "pressure_profile":           args.pressure_profile,
        "p_bottom_max":               args.p_bottom_max,
        "p_bottom_min":               args.p_bottom_min,
        "p_top_max":                  args.p_top_max,
        "p_top_min":                  args.p_top_min,
    }

    in_dir = Path(p["input_dir"])
    in_dir.mkdir(parents=True, exist_ok=True)

    # --- settings.json ---
    settings_path = in_dir / p["settings_file"]
    settings = build_settings(p)
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=4)
    print(f"[OK] settings written : {settings_path}")
    print(f"     solver output dir: {p['solver_output_dir']}  (created at runtime)")

    # --- input_edge_pressure.csv ---
    csv_path = in_dir / p["pressure_csv_file"]
    header, rows = build_pressure_header_and_rows(p)
    n = p["division_number"]
    dirichlet_count = sum(1 for e in EDGE_ORDER if p[f"bc_{e}"] == "dirichlet")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[OK] pressure CSV written: {csv_path}  "
          f"({len(rows)} timesteps, {dirichlet_count} Dirichlet edges × {n} = {dirichlet_count*n} columns)")


if __name__ == "__main__":
    main()
