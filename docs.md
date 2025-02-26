# HashDAG Project Documentation

## Project Overview
HashDAG is an implementation of a data structure for interactively modifying compressed sparse voxel geometry using directed acyclic graphs (DAGs). The project is based on research about efficiently editing compressed voxel representations while maintaining real-time performance. The core feature is the ability to perform interactive edits on high-resolution voxel scenes.

## Key Concepts
- **DAG (Directed Acyclic Graph)**: The primary data structure used for representing compressed voxel data
- **Hash Table**: Used for efficient node storage and lookup
- **Bloom Filters**: Used to accelerate node searches
- **CUDA Integration**: GPU-accelerated raytracing for real-time rendering

## Directory Structure

### Source Code (`src/`)
- **Main Entry Points**:
  - `main.cpp`: Application entry point
  - `engine.cpp/h`: Core engine managing DAGs, rendering, and user interaction

- **DAG Implementations** (`src/dags/`):
  - `basic_dag/`: Simple DAG implementation (likely for comparison)
  - `hash_dag/`: Main implementation with editing capabilities

- **Rendering**:
  - `dag_tracer.cu/h`: CUDA-based ray tracing for DAG visualization
  - `tracer.cu/h`: General raytracing functionality
  - `shaders/`: GLSL shaders for rendering

- **Utilities**:
  - `memory.cpp/h`: Memory management
  - `stats.cpp/h`: Performance statistics
  - `replay.cpp/h`: Replay functionality for reproducible testing

### Benchmarking (`python/`)
- Various benchmarking scripts for performance testing
- Tools for measuring and visualizing performance metrics

### Replays (`replays/`)
- Stored sequences of editing operations for testing and benchmarking

## Core Components

### Engine (`src/engine.cpp`, `src/engine.h`)
The central coordinator that manages the application lifecycle:
- **DAG Management**: Maintains DAG instances and handles switching between them
- **Rendering Pipeline**: Controls rendering settings and coordinates with DAGTracer
- **User Interaction**: Processes keyboard and mouse inputs for camera and editing
- **Editing System**: Implements the core editing loop that applies tools to the DAGs
- **Application Loop**: Manages the main execution flow with `loop()` and `tick()` functions
- **Camera Control**: Handles camera movement, rotation, and transitions
- **Replay System**: Provides recording and playback of editing sessions

Key functions include:
- `edit<T>()`: Template function for applying editing tools
- `tick()`: Core update function for processing inputs and updating the scene
- `loop()`: Main application loop
- `resolve_paths/colors/shadows()`: Functions that handle rendering phases

### Tracer and DAGTracer (`src/tracer.cu/h`, `src/dag_tracer.cu/h`)

#### DAGTracer
Acts as a bridge between CUDA/GPU operations and the main application:
- **Initialization**: Sets up CUDA-OpenGL interop for rendering
- **Path Resolution**: Maps rays from camera through the scene
- **Color Resolution**: Determines colors for visible voxels
- **Shadow Calculation**: Computes lighting effects
- **OpenGL Integration**: Renders the final image to screen

#### Tracer Namespace
Implements the core ray casting algorithms:
- **Ray-Based Algorithms**: Specialized algorithms for tracing rays through DAGs
- **DAG Traversal**: Efficient traversal for path finding and collision detection
- **Color Mapping**: Maps DAG data to colors for visualization 
- **Shadow Computation**: Implements ray-based shadow calculations

Core algorithms include:
- `trace_paths`: Casts primary rays to find visible voxels
- `trace_colors`: Determines colors for visible voxels 
- `trace_shadows`: Creates shadow effects
- `compute_intersection_mask`: Implements efficient DAG traversal

### HashDAG (`src/dags/hash_dag/hash_dag.h`)
The primary data structure supporting the compressed voxel representation:
- Manages the DAG structure for efficient voxel storage
- Interfaces with the hash table for node lookup
- Supports operations like add, remove, and modify voxels

### Hash Table (`src/dags/hash_dag/hash_table.h`, `hash_table.cpp`)
Core component that enables efficient node storage and retrieval:
- Uses bloom filters to speed up lookups
- Manages memory through a virtual page system
- Handles collisions and optimizes memory usage

### Editing System (`src/dags/hash_dag/hash_dag_edits.h`)
Provides functionality for modifying the DAG structure:
- Supports parallel editing operations
- Implements various editing tools (sphere, copy, etc.)
- Manages undo/redo functionality

### Editing Tools
The system implements multiple editing tools through a flexible `Editor` interface:

#### Editor Interface
- Base template class with derived editors for different operations
- `should_edit()`, `is_full()`, `is_empty()` methods determine editing scope
- `get_new_value()` and `get_new_color()` define how values are modified

#### Specific Editors
- **SphereEditor**: Adds/removes spherical regions of voxels
- **BoxEditor**: Adds/removes box-shaped regions
- **SpherePaintEditor**: Paints existing voxels without changing geometry
- **SphereNoiseEditor**: Adds procedural noise to spherical regions
- **CopyEditor**: Copies voxels from one location to another with optional transformations
- **FillEditor**: Implements flood-fill operations for connected regions

### Memory Management (`src/memory.cpp`, `src/memory.h`)
Custom memory handling for optimal performance:
- Page-based memory allocation
- Virtual memory mapping
- Efficient garbage collection

## Workflow

1. **Initialization**: The engine is initialized in `main.cpp` and loads DAG data
2. **User Interaction**: User inputs are processed by the `Engine` class
3. **Editing**: Edits are performed through the editing system in `hash_dag_edits.h`
4. **Rendering**: Results are visualized through the CUDA-based `DAGTracer`
5. **Replay/Benchmarking**: Performance can be measured using the benchmarking tools

## Performance Considerations
- Bloom filters accelerate node lookups
- Threaded editing operations enable parallelism
- CUDA-accelerated rendering provides real-time visualization
- Compact memory representation for efficient storage

## Additional Tools
- Benchmarking scripts in the `python/` directory
- Replay functionality for reproducing editing sequences
- Integration with Tracy profiler for performance analysis

## Extending the System

### Adding New Editing Tools
To add a new editing tool to the system:

1. Create a new class inheriting from `Editor<YourClass>` template
2. Implement core methods:
   - `should_edit_impl()`: Determines if editing applies to a region
   - `get_new_value()`: Defines geometry changes 
   - `get_new_color()`: Defines color changes
3. Add to the `ETool` enum in `tracer.h`
4. Add keyboard shortcut and tool handling in `Engine::key_callback_impl()`
5. Add case for the tool in the edit section of `Engine::tick()`

Example of minimal editor implementation:
```cpp
// New editor class
template<typename T>
class CustomEditor : public Editor<CustomEditor<T>> {
public:
    inline bool should_edit_impl(int32_t x, int32_t y, int32_t z) const {
        // Logic to determine if this voxel should be edited
        return condition;
    }
    
    inline T get_new_value(T old_val, int32_t x, int32_t y, int32_t z) const {
        // Logic to determine new voxel value
        return new_value;
    }
    
    inline color_t get_new_color(color_t old_color, int32_t x, int32_t y, int32_t z) const {
        // Logic to determine new color value
        return new_color;
    }
};
```

### Current Rendering Pipeline
The current rendering approach uses a multi-stage raytracing pipeline:

1. **Path Resolution**: `resolve_paths()` casts primary rays through the DAG
2. **Color Resolution**: `resolve_colors()` determines colors for visible voxels
3. **Shadow Calculation**: `resolve_shadows()` adds lighting effects
4. **Display**: Renders results to screen via OpenGL texture

### Implementing Alternative Rendering (Marching Cubes)
To implement marching cubes for smoother rendering:

1. **Extract Isosurface**: Use the DAG structure to determine where the isosurface should be
2. **Generate Mesh**: Implement the marching cubes algorithm to create triangle mesh
3. **Update Rendering Loop**: Modify `Engine::loop_graphics()` to handle mesh rendering
4. **Shader Implementation**: Create new shaders for mesh-based rendering

The system already has partial support for alternative rendering via `RenderMode::MarchingCubes`. Key components needed include:
- An isosurface extractor from the DAG data
- A mesh generator using marching cubes algorithm
- OpenGL vertex/index buffer setup
- Appropriate shaders for mesh rendering

Implementation notes:
- The DAG structure would need to be accessible for extracting the isosurface
- Real-time performance considerations are important for interactive editing
- The existing CUDA infrastructure could potentially be leveraged for acceleration
- Mesh generation would need to be performed after edits are applied to the DAG