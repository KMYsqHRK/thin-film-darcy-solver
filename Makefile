# Standalone Pressure Distribution Solver Makefile
# This builds the standalone version without ParticleWorks SDK dependencies

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I. -I./third_party -I/usr/include/eigen3
LDFLAGS =

# Directories
BIN_DIR = bin
OBJ_DIR = obj

# Source files
SOURCES = main.cpp thinfilm_solver.cpp
OBJECTS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
TARGET = $(BIN_DIR)/pressure_solver_standalone

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "Standalone solver built successfully: $(TARGET)"

# Compile source files
$(OBJ_DIR)/%.o: %.cpp thinfilm_solver.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create directories
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)


# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	rm -f output_*.csv pressure_grid.csv