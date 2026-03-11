#include <iostream>
#include "thinfilm_solver.hpp"

int main(int argc, char** argv)
{
    std::cout << "=== Standalone Pressure Distribution Solver ===" << std::endl;

    // Create solver instance
    standalone::pressure_solver::Solver solver;

    // Usage: ./pressure_solver_standalone [input_dir [output_dir]]
    //   input_dir : directory containing settings.json and input CSV (default: ../input)
    //   output_dir: directory for output files, created if absent  (default: ../output)
    std::string input_dir  = (argc > 1) ? argv[1] : "../input";
    std::string output_dir = (argc > 2) ? argv[2] : "../output";

    solver.load_settings(input_dir + "/settings.json", input_dir, output_dir);
    solver.load_input_data();

    // Process all timesteps
    solver.process_all_timesteps();

    // Save lift force output
    solver.save_lift_force_output();

    std::cout << "=== Processing Complete ===" << std::endl;
    return 0;
}
