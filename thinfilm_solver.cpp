#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include "nlohmann/single_include/nlohmann/json.hpp"
#include "thinfilm_solver.hpp"

using json = nlohmann::json;

namespace standalone { namespace pressure_solver {

// 設定読み込み
void Solver::load_settings(const std::string& filename)
{
    std::cout << "Loading settings from: " << filename << std::endl;

    // Initialize default settings
    m_settings.input_pressure_file = "input_edge_pressure.csv";
    m_settings.output_lift_force_file = "output_lift_forces.csv";
    m_settings.output_pressure_grid_file = "output_pressure_grid.csv";
    m_settings.division_number = 5;
    m_settings.hydraulic_conductivity = 1.0;
    m_settings.viscosity_coefficient = 1.0e-3;

    // Gravity defaults
    m_settings.gravity[0] = 0.0;
    m_settings.gravity[1] = 0.0;
    m_settings.gravity[2] = 9.8;

    // Rotation center defaults
    m_settings.rotation_center[0] = 0.0;
    m_settings.rotation_center[1] = 0.0;
    m_settings.rotation_center[2] = 0.0;

    // Output defaults
    m_settings.pressure_grid_save_interval = 10;
    m_settings.pressure_grid_include_wet_dry = true;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Cannot open settings file " << filename
                  << ". Using default settings." << std::endl;
        return;
    }

    try {
        json config;
        file >> config;

        // File settings
        if (config.contains("input_pressure_file")) {
            m_settings.input_pressure_file = config["input_pressure_file"];
        }

        if (config.contains("output_lift_force_file")) {
            m_settings.output_lift_force_file = config["output_lift_force_file"];
        }

        if (config.contains("output_pressure_grid_file")) {
            m_settings.output_pressure_grid_file = config["output_pressure_grid_file"];
        }

        // Division number
        if (config.contains("division_number")) {
            m_settings.division_number = config["division_number"];
        }

        // Boundary points
        if (config.contains("boundary_points")) {
            m_settings.boundary_points.clear();
            for (const auto& point_config : config["boundary_points"]) {
                BoundaryPoint point;
                point.name = point_config.value("name", "Point");

                if (point_config.contains("position")) {
                    point.x = point_config["position"].value("x", 0.0);
                    point.y = point_config["position"].value("y", 0.0);
                    point.z = point_config["position"].value("z", 0.0);
                }

                m_settings.boundary_points.push_back(point);
            }
        }

        // Gravity settings
        if (config.contains("gravity")) {
            const auto& gravity_config = config["gravity"];

            if (gravity_config.contains("x")) {
                m_settings.gravity[0] = gravity_config["x"];
            }

            if (gravity_config.contains("y")) {
                m_settings.gravity[1] = gravity_config["y"];
            }

            if (gravity_config.contains("z")) {
                m_settings.gravity[2] = gravity_config["z"];
            }
        }

        // Rotation center settings
        if (config.contains("rotation_center")) {
            const auto& rotation_center_config = config["rotation_center"];

            if (rotation_center_config.contains("x")) {
                m_settings.rotation_center[0] = rotation_center_config["x"];
            }

            if (rotation_center_config.contains("y")) {
                m_settings.rotation_center[1] = rotation_center_config["y"];
            }

            if (rotation_center_config.contains("z")) {
                m_settings.rotation_center[2] = rotation_center_config["z"];
            }
        }

        // Infiltration settings
        if (config.contains("infiltration")) {
            const auto& infiltration_config = config["infiltration"];
            if (infiltration_config.contains("hydraulic_conductivity")) {
                m_settings.hydraulic_conductivity = infiltration_config["hydraulic_conductivity"];
            }
            if (infiltration_config.contains("viscosity_coefficient")) {
                m_settings.viscosity_coefficient = infiltration_config["viscosity_coefficient"];
            }
        }

        // Pressure grid export settings
        if (config.contains("pressure_grid_export")) {
            const auto& pressure_grid_config = config["pressure_grid_export"];

            if (pressure_grid_config.contains("save_interval")) {
                m_settings.pressure_grid_save_interval = pressure_grid_config["save_interval"];
            }

            if (pressure_grid_config.contains("include_wet_dry")) {
                m_settings.pressure_grid_include_wet_dry = pressure_grid_config["include_wet_dry"];
            }
        }

        // Iterative solver settings
        if (config.contains("solver")) {
            const auto& solver_config = config["solver"];

            if (solver_config.contains("max_iterations")) {
                int config_max_iter = solver_config["max_iterations"];
                if (config_max_iter > 0 && config_max_iter <= 100000) {
                    m_max_iterations = config_max_iter;
                } else {
                    std::cerr << "Warning: Invalid max_iterations. Using default: " << m_max_iterations << std::endl;
                }
            }

            if (solver_config.contains("tolerance")) {
                double config_tol = solver_config["tolerance"];
                if (config_tol > 0.0 && config_tol < 1.0) {
                    m_solver_tolerance = config_tol;
                } else {
                    std::cerr << "Warning: Invalid tolerance. Using default: " << m_solver_tolerance << std::endl;
                }
            }
        }

        std::cout << "Settings loaded successfully:" << std::endl;
        std::cout << "  Input pressure file: " << m_settings.input_pressure_file << std::endl;
        std::cout << "  Output lift force file: " << m_settings.output_lift_force_file << std::endl;
        std::cout << "  Output pressure grid file: " << m_settings.output_pressure_grid_file << std::endl;
        std::cout << "  Division number: " << m_settings.division_number << std::endl;
        std::cout << "  Boundary points: " << m_settings.boundary_points.size() << std::endl;
        std::cout << "  Gravity: (" << m_settings.gravity[0] << ", " << m_settings.gravity[1] << ", "
                  << m_settings.gravity[2] << ") m/s^2" << std::endl;
        std::cout << "  Solver max iterations: " << m_max_iterations << std::endl;
        std::cout << "  Solver tolerance: " << m_solver_tolerance << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing settings file: " << e.what() << std::endl;
        std::cerr << "Using default settings." << std::endl;
    }
}

// Load time-series edge pressure data from CSV file
void Solver::load_input_data(const std::string& filename)
{
    std::cout << "Loading input pressure data from: " << filename << std::endl;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open input file " << filename << std::endl;
        return;
    }

    m_input_data.clear();
    std::string line;

    // Read header line
    if (!std::getline(file, line)) {
        std::cerr << "Error: Empty input file" << std::endl;
        return;
    }

    // Expected format: Time,P1,P2,P3,...,P(4n)
    // where 4n is the number of boundary points
    int n = m_settings.division_number;
    int expected_pressure_count = 4 * n;

    int line_number = 1;
    while (std::getline(file, line)) {
        line_number++;
        std::istringstream iss(line);
        std::string value;
        TimestepData data;

        // Read time
        if (!std::getline(iss, value, ',')) {
            std::cerr << "Warning: Skipping line " << line_number << " (missing time)" << std::endl;
            continue;
        }
        data.time = std::stod(value);

        // Read pressure values
        data.boundary_pressures.clear();
        while (std::getline(iss, value, ',')) {
            data.boundary_pressures.push_back(std::stod(value));
        }

        // Validate pressure count
        if (data.boundary_pressures.size() != static_cast<size_t>(expected_pressure_count)) {
            std::cerr << "Warning: Line " << line_number << " has " << data.boundary_pressures.size()
                      << " pressures, expected " << expected_pressure_count << std::endl;
            continue;
        }

        m_input_data.push_back(data);
    }

    file.close();
    std::cout << "Loaded " << m_input_data.size() << " timesteps from input file" << std::endl;
}

// 境界上の4n個の点を生成
std::vector<std::pair<double, double>> Solver::generate_boundary_points(int n, double deformation_x, double deformation_y)
{
    std::vector<std::pair<double, double>> boundary_points;
    
    if (m_settings.boundary_points.size() < 3) {
        std::cerr << "Error: Need at least 3 boundary points to define parallelogram" << std::endl;
        return boundary_points;
    }
    
    // 3つの境界点から矩形（平行四辺形）を定義
    double x1 = m_settings.boundary_points[0].x + deformation_x, y1 = m_settings.boundary_points[0].y + deformation_y;
    double x2 = m_settings.boundary_points[1].x + deformation_x, y2 = m_settings.boundary_points[1].y + deformation_y; 
    double x3 = m_settings.boundary_points[2].x + deformation_x, y3 = m_settings.boundary_points[2].y + deformation_y;
    
    // 4番目の点を計算（平行四辺形の対角点）
    double x4 = x2 + x3 - x1;
    double y4 = y2 + y3 - y1;
    
    // 各辺でn個の点を生成（4辺で4n個）
    // 辺1: (x1,y1) -> (x2,y2)
    for (int i = 0; i < n; i++) {
        double t = (double)i / n;
        double x = x1 + t * (x2 - x1);
        double y = y1 + t * (y2 - y1);
        boundary_points.push_back({x, y});
    }
    
    // 辺2: (x2,y2) -> (x4,y4)
    for (int i = 0; i < n; i++) {
        double t = (double)i / n;
        double x = x2 + t * (x4 - x2);
        double y = y2 + t * (y4 - y2);
        boundary_points.push_back({x, y});
    }
    
    // 辺3: (x4,y4) -> (x3,y3)
    for (int i = 0; i < n; i++) {
        double t = (double)i / n;
        double x = x4 + t * (x3 - x4);
        double y = y4 + t * (y3 - y4);
        boundary_points.push_back({x, y});
    }
    
    // 辺4: (x3,y3) -> (x1,y1)
    for (int i = 0; i < n; i++) {
        double t = (double)i / n;
        double x = x3 + t * (x1 - x3);
        double y = y3 + t * (y1 - y3);
        boundary_points.push_back({x, y});
    }
    
    return boundary_points;
}

// 境界の圧力値だけが格納された(n+1)×(n+1)の行列を作成
Eigen::MatrixXd Solver::create_pressure_matrix(const std::vector<double>& boundary_pressures, int n)
{
    Eigen::MatrixXd pressure_matrix = Eigen::MatrixXd::Zero(n + 1, n + 1);
    
    if (boundary_pressures.size() != 4 * n) {
        std::cerr << "Error: Expected " << 4 * n << " boundary pressure values, got " 
                  << boundary_pressures.size() << std::endl;
        return pressure_matrix;
    }
    
    // 境界値を行列の境界要素に配置
    int idx = 0;
    
    // 下辺 (i=0, j=0 to n)
    for (int j = 0; j < n; j++) {
        pressure_matrix(0, j) = boundary_pressures[idx++];
    }
    
    // 右辺 (i=0 to n, j=n)
    for (int i = 0; i < n; i++) {
        pressure_matrix(i, n) = boundary_pressures[idx++];
    }
    
    // 上辺 (i=n, j=n to 0)
    for (int j = n; j > 0; j--) {
        pressure_matrix(n, j) = boundary_pressures[idx++];
    }
    
    // 左辺 (i=n to 0, j=0)
    for (int i = n; i > 0; i--) {
        pressure_matrix(i, 0) = boundary_pressures[idx++];
    }
    
    return pressure_matrix;
}

// Update wet-dry matrix based on infiltration spreading (Darcy's law)
// Models infiltration spreading from each cell similar to Conway's Game of Life
void Solver::update_wet_dry_matrix(const std::vector<double>& boundary_pressures)
{
    std::cout << "Updating wet-dry matrix based on infiltration spreading (Darcy's law)" << std::endl;

    int n = m_settings.division_number;
    double lattice_x = std::abs(m_settings.boundary_points[1].x - m_settings.boundary_points[0].x);
    double lattice_y = std::abs(m_settings.boundary_points[2].y - m_settings.boundary_points[0].y);

    // Calculate cell dimensions
    double dx = lattice_x / n;  // Cell size in X direction
    double dy = lattice_y / n;  // Cell size in Y direction

    // Define threshold values
    const double eps = 1e-6;  // Minimum infiltration threshold
    double threshold_x = dx * dx;  // Threshold for X-direction spreading
    double threshold_y = dy * dy;  // Threshold for Y-direction spreading
    double threshold_diagonal = dx * dx + dy * dy;  // Threshold for diagonal spreading

    // Create a temporary matrix to store new wet cells (to avoid modifying during iteration)
    Eigen::MatrixXd new_wet_dry_matrix = m_wet_dry_matrix;

    // Process each grid cell
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            double infil_dist = m_grid_infil_distance(i, j);

            // Step 1: If infiltration distance exceeds eps, mark current cell as wet
            if (infil_dist > eps) {
                new_wet_dry_matrix(i, j) = 1.0;

                // Step 2: Check if we should propagate to X-direction neighbors
                if (infil_dist > threshold_x) {
                    // Set adjacent cells in X direction (i-1, i+1) to wet
                    if (i > 0) {
                        new_wet_dry_matrix(i - 1, j) = 1.0;  // South neighbor
                    }
                    if (i < n) {
                        new_wet_dry_matrix(i + 1, j) = 1.0;  // North neighbor
                    }
                }

                // Step 3: Check if we should propagate to Y-direction neighbors
                if (infil_dist > threshold_y) {
                    // Set adjacent cells in Y direction (j-1, j+1) to wet
                    if (j > 0) {
                        new_wet_dry_matrix(i, j - 1) = 1.0;  // West neighbor
                    }
                    if (j < n) {
                        new_wet_dry_matrix(i, j + 1) = 1.0;  // East neighbor
                    }
                }

                // Step 4: Check if we should propagate to diagonal neighbors
                if (infil_dist > threshold_diagonal) {
                    // Set diagonal neighbors to wet
                    if (i > 0 && j > 0) {
                        new_wet_dry_matrix(i - 1, j - 1) = 1.0;  // Southwest
                    }
                    if (i > 0 && j < n) {
                        new_wet_dry_matrix(i - 1, j + 1) = 1.0;  // Southeast
                    }
                    if (i < n && j > 0) {
                        new_wet_dry_matrix(i + 1, j - 1) = 1.0;  // Northwest
                    }
                    if (i < n && j < n) {
                        new_wet_dry_matrix(i + 1, j + 1) = 1.0;  // Northeast
                    }
                }
                // Once diagonal threshold is exceeded, we don't need to track further neighbors
                // (already handled above)
            }
        }
    }

    // Update the wet-dry matrix
    m_wet_dry_matrix = new_wet_dry_matrix;

    // Count wet points for logging
    int wet_count = 0;
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (m_wet_dry_matrix(i, j) > 0.5) {
                wet_count++;
            }
        }
    }

    std::cout << "Wet-dry matrix updated: " << wet_count << "/" << ((n+1)*(n+1)) << " points are wet" << std::endl;
    std::cout << "  Cell size: dx=" << dx << ", dy=" << dy << std::endl;
    std::cout << "  Thresholds: X=" << threshold_x << ", Y=" << threshold_y << ", Diagonal=" << threshold_diagonal << std::endl;
}

void Solver::update_infil_distance_matrix(const Eigen::MatrixXd& pressure_matrix, double dt)
{
    int n = m_settings.division_number;
    for (int i=0; i <= n; ++i){
        for (int j=0; j <= n; ++j){
            m_grid_infil_distance(i,j) += m_settings.hydraulic_conductivity * pressure_matrix(i,j) * dt / m_settings.viscosity_coefficient;
        }
    }
}

// Process all timesteps from input data
void Solver::process_all_timesteps()
{
    std::cout << "Processing " << m_input_data.size() << " timesteps..." << std::endl;

    int n = m_settings.division_number;
    double lattice_x = std::abs(m_settings.boundary_points[1].x - m_settings.boundary_points[0].x);
    double lattice_y = std::abs(m_settings.boundary_points[2].y - m_settings.boundary_points[0].y);

    // Initialize grid coordinates
    if (!m_grid_coords_initialized) {
        initialize_grid_coordinates(n, lattice_x, lattice_y);
    }

    // Initialize distance matrix
    if (!m_distance_matrix_initialized) {
        initialize_distance_matrix(n);
    }

    // Initialize wet-dry matrix
    if (!m_wet_dry_matrix_initialized) {
        initialize_wet_dry_matrix(n);
    }

    // Initialize infil distance matrix
    if (!m_grid_infil_distance_initialized){
        initialize_infil_distance_matrix(n);
    }

    // Clear output data
    m_output_times.clear();
    m_output_lift_forces.clear();
    m_output_torques.clear();

    // Process each timestep
    for (size_t i = 0; i < m_input_data.size(); ++i) {
        process_single_timestep(m_input_data[i], static_cast<int>(i));
    }

    std::cout << "All timesteps processed successfully" << std::endl;
}

// Process a single timestep
void Solver::process_single_timestep(const TimestepData& data, int timestep_index)
{
    int n = m_settings.division_number;
    double lattice_x = std::abs(m_settings.boundary_points[1].x - m_settings.boundary_points[0].x);
    double lattice_y = std::abs(m_settings.boundary_points[2].y - m_settings.boundary_points[0].y);

    std::cout << "Processing timestep " << timestep_index << " at time=" << data.time << std::endl;

    // Create pressure matrix from boundary pressures
    boundary_conditions = create_pressure_matrix(data.boundary_pressures, n);

    // Solve for pressure distribution
    Eigen::MatrixXd pressure_distribution = solve_pressure_distribution(boundary_conditions, n);

    // Calculate time step (assume uniform spacing, or could be stored in TimestepData)
    double dt = (timestep_index > 0) ? (data.time - m_input_data[timestep_index - 1].time) : 0.001;

    // Update infiltration distance matrix
    update_infil_distance_matrix(pressure_distribution, dt);

    // Update wet-dry matrix if needed
    int total_points = (n+1) * (n+1);
    int count_wet = (m_wet_dry_matrix.array() > 0.9).count();
    bool has_dry_points = (count_wet != total_points);
    bool has_pressure = !pressure_distribution.isZero(1e-6);

    if (has_pressure && has_dry_points) {
        update_wet_dry_matrix(data.boundary_pressures);
    }

    // Calculate lift force and torque from pressure distribution
    std::pair<double, double> lift_and_torque = calculate_lift_force_and_torque(pressure_distribution, lattice_x, lattice_y);

    // Store results
    m_output_times.push_back(data.time);
    m_output_lift_forces.push_back(lift_and_torque.first);
    m_output_torques.push_back(lift_and_torque.second);

    // Save pressure grid periodically
    if (timestep_index % m_settings.pressure_grid_save_interval == 0) {
        save_pressure_grid(pressure_distribution, m_wet_dry_matrix, data.time, timestep_index);
    }
}

// Creates coefficient matrix for solving Poisson equation with iterative solver
// Compressed system: builds matrix only for wet points with P=0 at wet-dry interface
Eigen::SparseMatrix<double> Solver::create_coefficient_matrix(int n, const Eigen::MatrixXd& wet_dry_matrix,
                                                               const Eigen::MatrixXi& grid_to_compressed)
{
    // Count wet points to determine matrix size
    int wet_point_count = 0;
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (grid_to_compressed(i, j) >= 0) {
                wet_point_count++;
            }
        }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(9 * wet_point_count);  // Estimate for 9-point stencil

    std::cout << "Building compressed coefficient matrix for " << wet_point_count << " wet points (9-point stencil)..." << std::endl;

    // Create coefficient matrix for 2D Poisson equation using finite differences
    // For each wet interior point, apply standard 9-point stencil
    // At wet-dry interfaces, apply P=0 Dirichlet boundary condition
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            // Skip dry points and original boundary points
            if (grid_to_compressed(i, j) < 0) continue;

            // Skip original boundary points (i=0, i=n, j=0, j=n)
            if (i == 0 || i == n || j == 0 || j == n) continue;

            int row = grid_to_compressed(i, j);
            double diagonal_value = 0.0;

            // 9-point stencil: check all 8 neighbors (4 direct + 4 diagonal)
            // Direct neighbors (N, S, E, W) and diagonal neighbors (NE, NW, SE, SW)

            // South neighbor (i-1, j)
            if (i > 0) {
                if (wet_dry_matrix(i-1, j) > 0.5 && grid_to_compressed(i-1, j) >= 0) {
                    // Wet neighbor - add to matrix
                    int col = grid_to_compressed(i-1, j);
                    triplets.emplace_back(row, col, 1.0);
                    diagonal_value -= 1.0;
                } else {
                    // Dry neighbor or original boundary - P=0 Dirichlet BC (no contribution)
                    diagonal_value -= 1.0;
                }
            }

            // North neighbor (i+1, j)
            if (i < n) {
                if (wet_dry_matrix(i+1, j) > 0.5 && grid_to_compressed(i+1, j) >= 0) {
                    // Wet neighbor - add to matrix
                    int col = grid_to_compressed(i+1, j);
                    triplets.emplace_back(row, col, 1.0);
                    diagonal_value -= 1.0;
                } else {
                    // Dry neighbor or original boundary - P=0 Dirichlet BC (no contribution)
                    diagonal_value -= 1.0;
                }
            }

            // West neighbor (i, j-1)
            if (j > 0) {
                if (wet_dry_matrix(i, j-1) > 0.5 && grid_to_compressed(i, j-1) >= 0) {
                    // Wet neighbor - add to matrix
                    int col = grid_to_compressed(i, j-1);
                    triplets.emplace_back(row, col, 1.0);
                    diagonal_value -= 1.0;
                } else {
                    // Dry neighbor or original boundary - P=0 Dirichlet BC (no contribution)
                    diagonal_value -= 1.0;
                }
            }

            // East neighbor (i, j+1)
            if (j < n) {
                if (wet_dry_matrix(i, j+1) > 0.5 && grid_to_compressed(i, j+1) >= 0) {
                    // Wet neighbor - add to matrix
                    int col = grid_to_compressed(i, j+1);
                    triplets.emplace_back(row, col, 1.0);
                    diagonal_value -= 1.0;
                } else {
                    // Dry neighbor or original boundary - P=0 Dirichlet BC (no contribution)
                    diagonal_value -= 1.0;
                }
            }

            // Southwest neighbor (i-1, j-1)
            if (i > 0 && j > 0) {
                if (wet_dry_matrix(i-1, j-1) > 0.5 && grid_to_compressed(i-1, j-1) >= 0) {
                    // Wet neighbor - add to matrix
                    int col = grid_to_compressed(i-1, j-1);
                    triplets.emplace_back(row, col, 1.0);
                    diagonal_value -= 1.0;
                } else {
                    // Dry neighbor or original boundary - P=0 Dirichlet BC (no contribution)
                    diagonal_value -= 1.0;
                }
            }

            // Southeast neighbor (i-1, j+1)
            if (i > 0 && j < n) {
                if (wet_dry_matrix(i-1, j+1) > 0.5 && grid_to_compressed(i-1, j+1) >= 0) {
                    // Wet neighbor - add to matrix
                    int col = grid_to_compressed(i-1, j+1);
                    triplets.emplace_back(row, col, 1.0);
                    diagonal_value -= 1.0;
                } else {
                    // Dry neighbor or original boundary - P=0 Dirichlet BC (no contribution)
                    diagonal_value -= 1.0;
                }
            }

            // Northwest neighbor (i+1, j-1)
            if (i < n && j > 0) {
                if (wet_dry_matrix(i+1, j-1) > 0.5 && grid_to_compressed(i+1, j-1) >= 0) {
                    // Wet neighbor - add to matrix
                    int col = grid_to_compressed(i+1, j-1);
                    triplets.emplace_back(row, col, 1.0);
                    diagonal_value -= 1.0;
                } else {
                    // Dry neighbor or original boundary - P=0 Dirichlet BC (no contribution)
                    diagonal_value -= 1.0;
                }
            }

            // Northeast neighbor (i+1, j+1)
            if (i < n && j < n) {
                if (wet_dry_matrix(i+1, j+1) > 0.5 && grid_to_compressed(i+1, j+1) >= 0) {
                    // Wet neighbor - add to matrix
                    int col = grid_to_compressed(i+1, j+1);
                    triplets.emplace_back(row, col, 1.0);
                    diagonal_value -= 1.0;
                } else {
                    // Dry neighbor or original boundary - P=0 Dirichlet BC (no contribution)
                    diagonal_value -= 1.0;
                }
            }

            // Add diagonal entry
            triplets.emplace_back(row, row, diagonal_value);
        }
    }

    // Build sparse matrix
    Eigen::SparseMatrix<double> A(wet_point_count, wet_point_count);
    A.setFromTriplets(triplets.begin(), triplets.end());

    std::cout << "Compressed coefficient matrix created: " << wet_point_count << "x" << wet_point_count
              << " system with " << triplets.size() << " non-zero entries" << std::endl;

    return A;
}

// Constructs the right-hand side vector from boundary conditions
// Compressed system: constructs RHS for wet points only with P=0 at wet-dry interface
Eigen::VectorXd Solver::construct_rhs_vector(const Eigen::MatrixXd& boundary_matrix, int n,
                                             const Eigen::MatrixXd& wet_dry_matrix,
                                             const Eigen::MatrixXi& grid_to_compressed)
{
    // Count wet points to determine vector size
    int wet_point_count = 0;
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (grid_to_compressed(i, j) >= 0) {
                wet_point_count++;
            }
        }
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(wet_point_count);

    std::cout << "Building compressed RHS vector for " << wet_point_count << " wet points (9-point stencil)..." << std::endl;

    // Fill RHS vector with boundary condition contributions
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            // Skip dry points and original boundary points
            if (grid_to_compressed(i, j) < 0) continue;
            if (i == 0 || i == n || j == 0 || j == n) continue;

            int idx = grid_to_compressed(i, j);

            // Add boundary contributions (negative because moved to RHS)
            // The boundary can be either:
            // 1. Original boundary (i=0, i=n, j=0, j=n) with measured pressure
            // 2. Wet-dry interface with P=0 (Dirichlet BC)

            // Direct neighbors (4-point contribution)

            // South neighbor (i-1, j)
            if (i > 0) {
                if (i-1 == 0) {
                    // Original boundary
                    rhs(idx) -= boundary_matrix(0, j);
                } else if (wet_dry_matrix(i-1, j) < 0.5 || grid_to_compressed(i-1, j) < 0) {
                    // Wet-dry interface: P=0 (no contribution to RHS)
                    rhs(idx) -= 0.0;
                }
            }

            // North neighbor (i+1, j)
            if (i < n) {
                if (i+1 == n) {
                    // Original boundary
                    rhs(idx) -= boundary_matrix(n, j);
                } else if (wet_dry_matrix(i+1, j) < 0.5 || grid_to_compressed(i+1, j) < 0) {
                    // Wet-dry interface: P=0 (no contribution to RHS)
                    rhs(idx) -= 0.0;
                }
            }

            // West neighbor (i, j-1)
            if (j > 0) {
                if (j-1 == 0) {
                    // Original boundary
                    rhs(idx) -= boundary_matrix(i, 0);
                } else if (wet_dry_matrix(i, j-1) < 0.5 || grid_to_compressed(i, j-1) < 0) {
                    // Wet-dry interface: P=0 (no contribution to RHS)
                    rhs(idx) -= 0.0;
                }
            }

            // East neighbor (i, j+1)
            if (j < n) {
                if (j+1 == n) {
                    // Original boundary
                    rhs(idx) -= boundary_matrix(i, n);
                } else if (wet_dry_matrix(i, j+1) < 0.5 || grid_to_compressed(i, j+1) < 0) {
                    // Wet-dry interface: P=0 (no contribution to RHS)
                    rhs(idx) -= 0.0;
                }
            }

            // Diagonal neighbors (4-point contribution for 9-point stencil)

            // Southwest neighbor (i-1, j-1)
            if (i > 0 && j > 0) {
                if ((i-1 == 0 || j-1 == 0)) {
                    // Original boundary
                    rhs(idx) -= boundary_matrix(i-1, j-1);
                } else if (wet_dry_matrix(i-1, j-1) < 0.5 || grid_to_compressed(i-1, j-1) < 0) {
                    // Wet-dry interface: P=0 (no contribution to RHS)
                    rhs(idx) -= 0.0;
                }
            }

            // Southeast neighbor (i-1, j+1)
            if (i > 0 && j < n) {
                if ((i-1 == 0 || j+1 == n)) {
                    // Original boundary
                    rhs(idx) -= boundary_matrix(i-1, j+1);
                } else if (wet_dry_matrix(i-1, j+1) < 0.5 || grid_to_compressed(i-1, j+1) < 0) {
                    // Wet-dry interface: P=0 (no contribution to RHS)
                    rhs(idx) -= 0.0;
                }
            }

            // Northwest neighbor (i+1, j-1)
            if (i < n && j > 0) {
                if ((i+1 == n || j-1 == 0)) {
                    // Original boundary
                    rhs(idx) -= boundary_matrix(i+1, j-1);
                } else if (wet_dry_matrix(i+1, j-1) < 0.5 || grid_to_compressed(i+1, j-1) < 0) {
                    // Wet-dry interface: P=0 (no contribution to RHS)
                    rhs(idx) -= 0.0;
                }
            }

            // Northeast neighbor (i+1, j+1)
            if (i < n && j < n) {
                if ((i+1 == n || j+1 == n)) {
                    // Original boundary
                    rhs(idx) -= boundary_matrix(i+1, j+1);
                } else if (wet_dry_matrix(i+1, j+1) < 0.5 || grid_to_compressed(i+1, j+1) < 0) {
                    // Wet-dry interface: P=0 (no contribution to RHS)
                    rhs(idx) -= 0.0;
                }
            }
        }
    }

    std::cout << "Compressed RHS vector constructed with " << wet_point_count << " elements" << std::endl;
    return rhs;
}

// Solves the Poisson equation to obtain pressure distribution of inner points using iterative solver
// Uses compressed system that only includes wet points with P=0 at wet-dry interface
Eigen::MatrixXd Solver::solve_pressure_distribution(const Eigen::MatrixXd& boundary_matrix, int n)
{
    std::cout << "========== Solving pressure distribution with compressed wet-only system ==========" << std::endl;

    // Step 1: Create renumbering map from (i,j) grid coordinates to compressed indices
    // -1 means point is excluded from compressed system (dry or original boundary)
    Eigen::MatrixXi grid_to_compressed = Eigen::MatrixXi::Constant(n + 1, n + 1, -1);

    int compressed_index = 0;

    // First pass: assign indices to wet interior points
    for (int i = 1; i < n; ++i) {  // Skip original boundaries (i=0, i=n)
        for (int j = 1; j < n; ++j) {  // Skip original boundaries (j=0, j=n)
            if (m_wet_dry_matrix(i, j) > 0.5) {  // Wet point
                grid_to_compressed(i, j) = compressed_index++;
            }
        }
    }

    int total_wet_points = compressed_index;
    std::cout << "Grid renumbering complete: " << total_wet_points << " wet interior points" << std::endl;
    std::cout << "Total grid points (including boundaries): " << (n+1)*(n+1) << std::endl;
    std::cout << "Compression ratio: " << (double)total_wet_points / ((n+1)*(n+1)) * 100.0 << "%" << std::endl;

    // Handle edge case: if no wet points, return zero matrix
    if (total_wet_points == 0) {
        std::cout << "Warning: No wet interior points found. Returning zero pressure matrix." << std::endl;
        Eigen::MatrixXd pressure_matrix = Eigen::MatrixXd::Zero(n + 1, n + 1);
        // Keep original boundary values
        pressure_matrix = boundary_matrix;
        // Set all interior points to 0
        for (int i = 1; i < n; ++i) {
            for (int j = 1; j < n; ++j) {
                pressure_matrix(i, j) = 0.0;
            }
        }
        return pressure_matrix;
    }

    // Step 2: Create compressed coefficient matrix for wet points only
    std::cout << "Creating compressed coefficient matrix for iterative solver..." << std::endl;
    Eigen::SparseMatrix<double> A = create_coefficient_matrix(n, m_wet_dry_matrix, grid_to_compressed);

    // Step 3: Construct compressed RHS vector with P=0 at wet-dry interface
    Eigen::VectorXd rhs = construct_rhs_vector(boundary_matrix, n, m_wet_dry_matrix, grid_to_compressed);

    // Step 4: Configure and solve iterative system
    std::cout << "Configuring BiCGSTAB iterative solver..." << std::endl;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;

    solver.setMaxIterations(m_max_iterations);
    solver.setTolerance(m_solver_tolerance);

    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Error: Failed to initialize iterative solver" << std::endl;
        return boundary_matrix;  // Return boundary matrix as fallback
    }

    std::cout << "Solving compressed linear system (" << total_wet_points << " unknowns)..." << std::endl;
    Eigen::VectorXd solution = solver.solve(rhs);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Error: Iterative solver failed to converge" << std::endl;
        std::cerr << "  Iterations: " << solver.iterations() << std::endl;
        std::cerr << "  Estimated error: " << solver.error() << std::endl;
        return boundary_matrix;  // Return boundary matrix as fallback
    }

    std::cout << "Iterative solver converged successfully:" << std::endl;
    std::cout << "  Iterations: " << solver.iterations() << "/" << m_max_iterations << std::endl;
    std::cout << "  Estimated error: " << solver.error() << " (tolerance: " << m_solver_tolerance << ")" << std::endl;

    // Step 5: Map compressed solution back to full grid
    std::cout << "Mapping compressed solution back to full grid..." << std::endl;
    Eigen::MatrixXd pressure_matrix = boundary_matrix;  // Start with boundary values

    // Fill wet interior points with solution
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < n; ++j) {
            int compressed_idx = grid_to_compressed(i, j);
            if (compressed_idx >= 0) {
                // Wet point - use solved pressure
                pressure_matrix(i, j) = solution(compressed_idx);
            } else {
                // Dry point - P=0 (atmospheric pressure)
                pressure_matrix(i, j) = 0.0;
            }
        }
    }

    std::cout << "Pressure distribution solved successfully with physically correct wet-dry treatment" << std::endl;
    std::cout << "==============================================================================\n" << std::endl;

    return pressure_matrix;
}

// Calculates lift force and torque from pressure distribution
std::pair<double, double> Solver::calculate_lift_force_and_torque(const Eigen::MatrixXd& pressure_matrix, double length_x, double length_y)
{
    double total_force = 0.0;
    double total_torque = 0.0;
    double dx = length_x / (pressure_matrix.cols() - 1);
    double dy = length_y / (pressure_matrix.rows() - 1);
    double area_element = dx * dy;
    
    int rows = pressure_matrix.rows();
    int cols = pressure_matrix.cols();
    
    // Integrate pressure over the domain using trapezoidal rule
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double weight = 1.0;
            
            // Apply trapezoidal rule weights
            // Corner points get weight 0.25, edge points get weight 0.5, interior points get weight 1.0
            if ((i == 0 || i == rows - 1) && (j == 0 || j == cols - 1)) {
                weight = 0.25;  // Corner points
            } else if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                weight = 0.5;   // Edge points
            }
            
            total_force += pressure_matrix(i, j) * area_element * weight;
            total_torque += pressure_matrix(i, j) * area_element * weight * m_distance_matrix(i, j);
        }
    }
    
    std::cout << "Calculated upward resultant force: " << total_force << " N" << std::endl;
    return std::make_pair(total_force, total_torque);
}

// Save lift force output to CSV file
void Solver::save_lift_force_output() const
{
    std::ofstream file(m_settings.output_lift_force_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << m_settings.output_lift_force_file << " for writing" << std::endl;
        return;
    }

    // Write header
    file << "Time,LiftForce,Torque" << std::endl;

    // Write data
    for (size_t i = 0; i < m_output_times.size(); ++i) {
        file << m_output_times[i] << ","
             << m_output_lift_forces[i] << ","
             << m_output_torques[i] << std::endl;
    }

    file.close();
    std::cout << "Lift force output saved to " << m_settings.output_lift_force_file
              << " with " << m_output_times.size() << " data points" << std::endl;
}

void Solver::save_pressure_grid(const Eigen::MatrixXd& pressure_distribution,
                                const Eigen::MatrixXd& wet_dry_matrix,
                                double current_time,
                                int timestep_index) const
{
    // Check if file exists to determine if we need to write header
    bool file_exists = std::ifstream(m_settings.output_pressure_grid_file).good();

    // Open file in append mode
    std::ofstream file(m_settings.output_pressure_grid_file, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << m_settings.output_pressure_grid_file << " for writing" << std::endl;
        return;
    }

    // Write header if file is new
    if (!file_exists) {
        file << "Time,i,j,x,y,z,Pressure";
        if (m_settings.pressure_grid_include_wet_dry) {
            file << ",WetDry";
        }
        file << std::endl;
    }

    // Get grid parameters
    int n = m_settings.division_number;
    int grid_size = n + 1;  // (n+1) x (n+1) grid

    // Get boundary coordinates to calculate grid coordinates
    const auto& boundary_points = m_settings.boundary_points;
    if (boundary_points.size() < 3) {
        std::cerr << "Error: Not enough boundary points to calculate grid coordinates" << std::endl;
        return;
    }

    // Calculate lattice dimensions from boundary points
    // Point1 and Point2 define x-direction, Point1 and Point3 define y-direction
    double origin_x = boundary_points[0].x;
    double origin_y = boundary_points[0].y;
    double origin_z = boundary_points[0].z;

    double lattice_x = std::sqrt(std::pow(boundary_points[1].x - boundary_points[0].x, 2) +
                                  std::pow(boundary_points[1].y - boundary_points[0].y, 2));
    double lattice_y = std::sqrt(std::pow(boundary_points[2].x - boundary_points[0].x, 2) +
                                  std::pow(boundary_points[2].y - boundary_points[0].y, 2));

    double dx = lattice_x / n;
    double dy = lattice_y / n;

    // Write grid data
    int points_written = 0;
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            // Calculate grid coordinates
            double x = origin_x + j * dx;
            double y = origin_y + i * dy;
            double z = origin_z;

            // Get pressure value
            double pressure = pressure_distribution(i, j);

            // Write data line
            file << current_time << ","
                 << i << ","
                 << j << ","
                 << x << ","
                 << y << ","
                 << z << ","
                 << pressure;

            // Write wet/dry status if requested
            if (m_settings.pressure_grid_include_wet_dry) {
                double wet_dry = wet_dry_matrix(i, j);
                file << "," << wet_dry;
            }

            file << std::endl;
            points_written++;
        }
    }

    file.close();
    std::cout << "Pressure grid saved to " << m_settings.output_pressure_grid_file
              << " at time " << current_time << " (timestep " << timestep_index << ", "
              << points_written << " points)" << std::endl;
}

// Calculate (n+1)×(n+1) grid coordinates representing pressure distribution points
void Solver::initialize_grid_coordinates(int n, double lattice_x, double lattice_y)
{
    // Create 3D coordinate matrix: (n+1)×(n+1)×3 where last dimension is [x,y,z]
    m_grid_coords.resize(n + 1, (n + 1) * 3);

    if (m_settings.boundary_points.size() < 3) {
        std::cerr << "Error: Need at least 3 boundary points to calculate grid coordinates" << std::endl;
    }

    // Use the first boundary point as the origin reference
    double origin_x = m_settings.boundary_points[0].x;
    double origin_y = m_settings.boundary_points[0].y;
    double origin_z = m_settings.boundary_points[0].z;

    // Calculate grid spacing
    double dx = lattice_x / n;
    double dy = lattice_y / n;

    // Generate grid coordinates
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            double x = origin_x + j * dx;
            double y = origin_y + i * dy;
            double z = origin_z;  // Assuming constant z-level

            // Store coordinates in flattened format: [x1,y1,z1,x2,y2,z2,...]
            int col_base = j * 3;
            m_grid_coords(i, col_base) = x;
            m_grid_coords(i, col_base + 1) = y;
            m_grid_coords(i, col_base + 2) = z;
        }
    }
    m_grid_coords_initialized = true;
    std::cout << "Grid coordinates calculated for " << (n+1) << "×" << (n+1) << " points" << std::endl;
}

// Initialize distance matrix from rotation center to each pressure measurement point
void Solver::initialize_distance_matrix(int n)
{
    if (m_distance_matrix_initialized && m_cached_n == n) {
        std::cout << "Distance matrix already initialized for n=" << n << std::endl;
        return;
    }

    double lattice_x = std::abs(m_settings.boundary_points[1].x - m_settings.boundary_points[0].x);
    double lattice_y = std::abs(m_settings.boundary_points[2].y - m_settings.boundary_points[0].y);

    // Initialize distance matrix
    m_distance_matrix = Eigen::MatrixXd::Zero(n + 1, n + 1);

    double rotation_center_x = m_settings.rotation_center[0];
    double rotation_center_y = m_settings.rotation_center[1];
    double rotation_center_z = m_settings.rotation_center[2];

    // Calculate distances from rotation center to each grid point
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            int col_base = j * 3;
            double point_x = m_grid_coords(i, col_base);
            double point_y = m_grid_coords(i, col_base + 1);
            double point_z = m_grid_coords(i, col_base + 2);

            // Calculate 3D Euclidean distance
            double dx = point_x - rotation_center_x;
            double dy = point_y - rotation_center_y;
            double dz = point_z - rotation_center_z;
            double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

            m_distance_matrix(i, j) = distance;
        }
    }

    m_distance_matrix_initialized = true;
    std::cout << "Distance matrix initialized successfully for " << (n+1) << "×" << (n+1) << " grid" << std::endl;
    std::cout << "Rotation center: (" << rotation_center_x << ", " << rotation_center_y << ", " << rotation_center_z << ")" << std::endl;
}

// Initialize wet-dry matrix (1 = wet, 0 = dry)
void Solver::initialize_wet_dry_matrix(int n)
{
    if (m_wet_dry_matrix_initialized && m_cached_n == n) {
        std::cout << "Wet-dry matrix already initialized for n=" << n << std::endl;
        return;
    }

    // Initialize with all points as dry (value = 0.0)
    m_wet_dry_matrix = Eigen::MatrixXd::Zero(n + 1, n + 1);
    m_wet_dry_matrix_initialized = true;
    std::cout << "Wet-dry matrix initialized successfully for " << (n+1) << "×" << (n+1) << " grid (all dry)" << std::endl;
}

void Solver::initialize_infil_distance_matrix(int n)
{
    if (m_grid_infil_distance_initialized && m_cached_n == n){
        std::cout << "grid infil distance matrix already initialized for n=" << n << std::endl;
        return;
    }
    m_grid_infil_distance = Eigen::MatrixXd::Zero(n + 1, n + 1);
    m_grid_infil_distance_initialized = true;
    std::cout << "grid infil distance matrix initialized successfully for " << (n+1) << "×" << (n+1) << " grid (all dry)" << std::endl;
}

}}  