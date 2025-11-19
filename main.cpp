#include <iostream>
#include "thinfilm_solver.hpp"

int main(int argc, char** argv)
{
    std::cout << "=== Standalone Pressure Distribution Solver ===" << std::endl;

    // Create solver instance
    standalone::pressure_solver::Solver solver;

    // Load settings from JSON file
    std::string settings_file = (argc > 1) ? argv[1] : "../input/settings.json";
    solver.load_settings(settings_file);

    // Load input edge pressure data
    std::string input_file = (argc > 2) ? argv[2] : "../input/input_edge_pressure.csv";
    solver.load_input_data(input_file);

    // Process all timesteps
    solver.process_all_timesteps();

    // Save lift force output
    solver.save_lift_force_output();

    std::cout << "=== Processing Complete ===" << std::endl;
    return 0;
}
