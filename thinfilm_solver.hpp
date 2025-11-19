#pragma once
#if !defined(STANDALONE_PRESSURE_SOLVER_H_INCLUDED)
#define STANDALONE_PRESSURE_SOLVER_H_INCLUDED

#include <string>
#include <vector>
#include <eigen/Eigen/Sparse>

namespace standalone { namespace pressure_solver {

// 境界点の設定
struct BoundaryPoint {
    std::string name;
    double x, y, z;
};

// Time-series data for a single timestep
struct TimestepData {
    double time;
    std::vector<double> boundary_pressures;  // 4n boundary pressure values
};

// 設定データ
struct Settings {
    // Input/Output files
    std::string input_pressure_file;         // CSV file with time-series edge pressure data
    std::string output_lift_force_file;      // CSV file for lift force time-series
    std::string output_pressure_grid_file;   // CSV file for pressure distribution

    // Geometry parameters
    int division_number;
    std::vector<BoundaryPoint> boundary_points;

    // Solver parameters
    double hydraulic_conductivity;            // Darcy's law permeability coefficient
    double kinematic_viscosity_coefficient;   // Kinematic viscosity coefficient

    double gravity[3];                        // Gravity vector
    double rotation_center[3];                // Rotation center for torque calculation

    // Output control
    int pressure_grid_save_interval;
    bool pressure_grid_include_wet_dry;
};

// Standalone solver module
class Solver {
private:
    Settings m_settings;

    // Iterative solver settings
    int m_max_iterations;
    double m_solver_tolerance;
    int m_cached_n;

    // Grid coordinates for pressure points
    Eigen::MatrixXd m_grid_coords;
    bool m_grid_coords_initialized;

    // Distance matrix from rotation center to pressure points
    Eigen::MatrixXd m_distance_matrix;
    bool m_distance_matrix_initialized;

    // Wet-dry state matrix
    Eigen::MatrixXd m_wet_dry_matrix;
    bool m_wet_dry_matrix_initialized;

    // Infiltration distance matrix
    Eigen::MatrixXd m_grid_infil_distance;
    bool m_grid_infil_distance_initialized;

    // Time-series input data
    std::vector<TimestepData> m_input_data;

    // Output data vectors
    std::vector<double> m_output_times;
    std::vector<double> m_output_lift_forces;
    std::vector<double> m_output_torques;

public:
    // Constructor
    Solver() : m_max_iterations(1000), m_solver_tolerance(1e-6), m_cached_n(0),
               m_grid_coords_initialized(false), m_distance_matrix_initialized(false),
               m_wet_dry_matrix_initialized(false), m_grid_infil_distance_initialized(false) {}

    // members
    Eigen::MatrixXd boundary_conditions;

    // Main public interface
    void load_settings(const std::string& filename = "settings.json");
    void load_input_data(const std::string& filename);
    void process_all_timesteps();
    void save_lift_force_output() const;

    // Initialization methods
    void initialize_distance_matrix(int n);
    void initialize_wet_dry_matrix(int n);
    void initialize_grid_coordinates(int n, double lattice_x, double lattice_y);
    void initialize_infil_distance_matrix(int n);

    // Core solver methods
    Eigen::SparseMatrix<double> create_coefficient_matrix(int n, const Eigen::MatrixXd& wet_dry_matrix,
                                                          const Eigen::MatrixXi& grid_to_compressed);
    Eigen::VectorXd construct_rhs_vector(const Eigen::MatrixXd& boundary_matrix, int n,
                                         const Eigen::MatrixXd& wet_dry_matrix,
                                         const Eigen::MatrixXi& grid_to_compressed);
    Eigen::MatrixXd solve_pressure_distribution(const Eigen::MatrixXd& boundary_matrix, int n);

    // Boundary condition methods
    std::vector<std::pair<double, double>> generate_boundary_points(int n, double deformation_x, double deformation_y);
    Eigen::MatrixXd create_pressure_matrix(const std::vector<double>& boundary_pressures, int n);

    // Infiltration and wet-dry update
    void update_infil_distance_matrix(const Eigen::MatrixXd& pressure_matrix, double dt);
    void update_wet_dry_matrix(const std::vector<double>& boundary_pressures);

    // Force calculation
    std::pair<double, double> calculate_lift_force_and_torque(const Eigen::MatrixXd& pressure_matrix,
                                                               double length_x, double length_y);

    // Output methods
    void save_pressure_grid(const Eigen::MatrixXd& pressure_distribution,
                           const Eigen::MatrixXd& wet_dry_matrix,
                           double current_time,
                           int timestep_index) const;

private:
    void process_single_timestep(const TimestepData& data, int timestep_index);
};

}}
#endif