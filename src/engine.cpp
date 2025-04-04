﻿#include "engine.h"
#include "hacky_profiler.hpp"
#include "shader.h"
#include "utils.h"
#include "marching_cubes_tables.h"
#include "memory.h"
#include "dags/hash_dag/hash_dag_editors.h"
#include <math.h>

#include "glfont.h"

//-----------------------------------------------------------------------------
// Engine Singleton Initialization
//-----------------------------------------------------------------------------
Engine Engine::engine;

//-----------------------------------------------------------------------------
// Helper Functions
//-----------------------------------------------------------------------------
/// Clear the console screen
inline void clear_console()
{
    printf("\033[H\033[J");
}

/// Convert a DAG type to its string representation
std::string dag_to_string(EDag dag)
{
    switch (dag)
    {
    case EDag::BasicDagUncompressedColors:
        return "BasicDagUncompressedColors";
    case EDag::BasicDagCompressedColors:
        return "BasicDagCompressedColors";
    case EDag::BasicDagColorErrors:
        return "BasicDagColorErrors";
    case EDag::HashDag:
        return "HashDag";
    default:
        check(false);
        return "";
    }
}

/// Convert a tool type to its string representation
std::string tool_to_string(ETool tool)
{
    switch (tool)
    {
    case ETool::Sphere:
        return "Sphere";
    case ETool::SpherePaint:
        return "SpherePaint";
        ;
    case ETool::SphereNoise:
        return "SphereNoise";
    case ETool::Cube:
        return "Cube";
    case ETool::CubeCopy:
        return "CubeCopy";
    case ETool::CubeFill:
        return "CubeFill";
    default:
        check(false);
        return "";
    }
}

//-----------------------------------------------------------------------------
// DAG Management
//-----------------------------------------------------------------------------
/// Check if a DAG type is valid (has been initialized)
bool Engine::is_dag_valid(EDag dag) const
{
    switch (dag)
    {
    case EDag::BasicDagUncompressedColors:
        return basicDag.is_valid() && basicDagUncompressedColors.is_valid();
    case EDag::BasicDagCompressedColors:
        return basicDag.is_valid() && basicDagCompressedColors.is_valid();
    case EDag::BasicDagColorErrors:
        return basicDag.is_valid() && basicDagColorErrors.is_valid();
    case EDag::HashDag:
        return hashDag.is_valid() && hashDagColors.is_valid();
    default:
        check(false);
        return false;
    }
}

/// Move to the next valid DAG type
void Engine::next_dag()
{
    do
    {
        config.currentDag = EDag((uint32(config.currentDag) + 1) % CNumDags);
    } while (!is_dag_valid(config.currentDag));
}

/// Move to the previous valid DAG type
void Engine::previous_dag()
{
    do
    {
        config.currentDag = EDag(Utils::subtract_mod(uint32(config.currentDag), CNumDags));
    } while (!is_dag_valid(config.currentDag));
}

/// Set the current DAG type, ensuring it's valid
void Engine::set_dag(EDag dag)
{
    config.currentDag = dag;
    if (!is_dag_valid(config.currentDag))
    {
        next_dag();
    }
}

//-----------------------------------------------------------------------------
// Input Handling
//-----------------------------------------------------------------------------
/// Handle keyboard input events
void Engine::key_callback_impl(int key, int scancode, int action, int mods)
{
    if (!((0 <= key) && (key <= GLFW_KEY_LAST))) // Media keys
    {
        return;
    }

    if (action == GLFW_RELEASE)
        state.keys[(uint64)key] = false;
    if (action == GLFW_PRESS)
        state.keys[(uint64)key] = true;

    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        if (key == GLFW_KEY_T)
        {
            renderMode = (renderMode == RenderMode::Default) ? RenderMode::MarchingCubes : RenderMode::Default;
            printf("Render Mode switched to: %s\n",
                   (renderMode == RenderMode::Default) ? "Default" : "MarchingCubes");
        }
        if (key == GLFW_KEY_M)
        {
            printMemoryStats = !printMemoryStats;
        }
#if UNDO_REDO
        if (key == GLFW_KEY_Z)
        {
            if (config.currentDag == EDag::HashDag)
            {
                if (state.keys[GLFW_KEY_LEFT_CONTROL] || state.keys[GLFW_KEY_RIGHT_CONTROL])
                {
                    if (state.keys[GLFW_KEY_LEFT_SHIFT])
                    {
                        undoRedo.redo(hashDag, hashDagColors);
                        replayWriter.add_action<ReplayActionRedo>();
                    }
                    else
                    {
                        undoRedo.undo(hashDag, hashDagColors);
                        replayWriter.add_action<ReplayActionUndo>();
                    }
                }
            }
        }
#endif
        if (key == GLFW_KEY_BACKSPACE)
        {
            replayWriter.write_csv();
            replayWriter.clear();
            printf("Replay saved!\n");
        }
        if (key == GLFW_KEY_R)
        {
            if (state.keys[GLFW_KEY_LEFT_SHIFT])
            {
                printf("Replay reader cleared\n");
                printf("Replay writer cleared\n");
                replayReader.clear();
                replayWriter.clear();
            }
            else
            {
                printf("Replay reader reset\n");
                printf("Stats cleared\n");
                statsRecorder.clear();
                replayReader.reset_replay();
            }
        }
        if (key == GLFW_KEY_TAB)
        {
            if (state.keys[GLFW_KEY_LEFT_SHIFT])
            {
                config.tool = ETool(Utils::subtract_mod(uint32(config.tool), CNumTools));
            }
            else
            {
                config.tool = ETool((uint32(config.tool) + 1) % CNumTools);
            }

            const auto str = tool_to_string(config.tool);
            printf("Current tool: %s\n", str.c_str());
        }
        if (key == GLFW_KEY_G)
        {
            if (config.currentDag == EDag::HashDag)
            {
                hashDag.remove_stale_nodes(hashDag.levels - 2);
            }
            undoRedo.free();
        }
        if (key == GLFW_KEY_C)
        {
            if (state.keys[GLFW_KEY_LEFT_SHIFT])
            {
                config.debugColors = EDebugColors(Utils::subtract_mod(uint32(config.debugColors), CNumDebugColors));
            }
            else
            {
                config.debugColors = EDebugColors((uint32(config.debugColors) + 1) % CNumDebugColors);
            }
        }
        if (key == GLFW_KEY_U)
        {
            auto previousGPUUsage = Memory::get_gpu_allocated_memory();
            auto previousCPUUsage = Memory::get_cpu_allocated_memory();
            undoRedo.free();
            printf("Undo redo cleared! Memory saved: GPU: %fMB CPU: %fMB\n",
                   Utils::to_MB(previousGPUUsage - Memory::get_gpu_allocated_memory()),
                   Utils::to_MB(previousCPUUsage - Memory::get_cpu_allocated_memory()));
        }
        if (key == GLFW_KEY_CAPS_LOCK)
        {
            if (state.keys[GLFW_KEY_LEFT_SHIFT])
            {
                previous_dag();
            }
            else
            {
                next_dag();
            }
            const auto str = dag_to_string(config.currentDag);
            printf("Current dag: %s\n", str.c_str());
        }
        if (key == GLFW_KEY_1)
        {
            config.debugColorsIndexLevel++;
            config.debugColorsIndexLevel = std::min(config.debugColorsIndexLevel, basicDag.levels);
        }
        if (key == GLFW_KEY_2)
        {
            config.debugColorsIndexLevel = uint32(std::max(int32(config.debugColorsIndexLevel) - 1, 0));
        }
        if (key == GLFW_KEY_3)
        {
            config.debugColors = EDebugColors::Index;
        }
        if (key == GLFW_KEY_4)
        {
            config.debugColors = EDebugColors::Position;
        }
        if (key == GLFW_KEY_5)
        {
            config.debugColors = EDebugColors::ColorTree;
        }
        if (key == GLFW_KEY_6)
        {
            config.debugColors = EDebugColors::ColorBits;
        }
        if (key == GLFW_KEY_7)
        {
            config.debugColors = EDebugColors::MinColor;
        }
        if (key == GLFW_KEY_8)
        {
            config.debugColors = EDebugColors::MaxColor;
        }
        if (key == GLFW_KEY_9)
        {
            config.debugColors = EDebugColors::Weight;
        }
        if (key == GLFW_KEY_0)
        {
            config.debugColors = EDebugColors::None;
        }
        if (key == GLFW_KEY_X)
        {
            shadows = !shadows;
        }
        if (key == GLFW_KEY_EQUAL)
        {
            shadowBias += 0.1f;
            printf("Shadow bias: %f\n", shadowBias);
        }
        if (key == GLFW_KEY_MINUS)
        {
            shadowBias -= 0.1f;
            printf("Shadow bias: %f\n", shadowBias);
        }
        if (key == GLFW_KEY_O)
        {
            fogDensity += 1;
            printf("Fog density: %f\n", fogDensity);
        }
        if (key == GLFW_KEY_H)
        {
            showUI = !showUI;
        }
        if (key == GLFW_KEY_F)
        {
            toggle_fullscreen();
        }

        const double rotationStep = (state.keys[GLFW_KEY_LEFT_SHIFT] || state.keys[GLFW_KEY_RIGHT_SHIFT] ? -10 : 10);
        if (key == GLFW_KEY_F1)
        {
            transformRotation.X += rotationStep;
            if (transformRotation.X > 180)
                transformRotation.X -= 360;
            if (transformRotation.X < -180)
                transformRotation.X += 360;
        }
        if (key == GLFW_KEY_F2)
        {
            transformRotation.Y += rotationStep;
            if (transformRotation.Y > 180)
                transformRotation.Y -= 360;
            if (transformRotation.Y < -180)
                transformRotation.Y += 360;
        }
        if (key == GLFW_KEY_F3)
        {
            transformRotation.Z += rotationStep;
            if (transformRotation.Z > 180)
                transformRotation.Z -= 360;
            if (transformRotation.Z < -180)
                transformRotation.Z += 360;
        }
        if (key == GLFW_KEY_F6)
        {
            transformScale += state.keys[GLFW_KEY_LEFT_SHIFT] || state.keys[GLFW_KEY_RIGHT_SHIFT] ? -.1f : .1f;
        }

        if (key == GLFW_KEY_F4)
        {
            enableSwirl = !enableSwirl;
        }
        if (key == GLFW_KEY_F5)
        {
            swirlPeriod += state.keys[GLFW_KEY_LEFT_SHIFT] || state.keys[GLFW_KEY_RIGHT_SHIFT] ? -10 : 10;
        }

        if (key == GLFW_KEY_I)
        {
            fogDensity -= 1;
            printf("Fog density: %f\n", fogDensity);
        }
        if (key == GLFW_KEY_P)
        {
            const bool printGlobalStats = state.keys[GLFW_KEY_LEFT_SHIFT];
            if (config.currentDag == EDag::BasicDagUncompressedColors)
            {
                if (printGlobalStats)
                    DAGUtils::print_stats(basicDag);
                basicDag.print_stats();
                basicDagUncompressedColors.print_stats();
            }
            else if (config.currentDag == EDag::BasicDagCompressedColors)
            {
                if (printGlobalStats)
                    DAGUtils::print_stats(basicDag);
                basicDag.print_stats();
                basicDagCompressedColors.print_stats();
            }
            else if (config.currentDag == EDag::BasicDagColorErrors)
            {
                if (printGlobalStats)
                    DAGUtils::print_stats(basicDag);
                basicDag.print_stats();
            }
            else if (config.currentDag == EDag::HashDag)
            {
                if (printGlobalStats)
                    DAGUtils::print_stats(hashDag);
                hashDag.data.print_stats();
                hashDagColors.print_stats();
#if UNDO_REDO
                undoRedo.print_stats();
#endif
            }
        }
        if (key == GLFW_KEY_L)
        {
            hashDag.data.save_bucket_sizes(false);
        }
        if (key == GLFW_KEY_KP_ENTER)
        {
            printf("view.rotation = { %f, %f, %f, %f, %f, %f, %f, %f, %f };\n",
                   view.rotation.D00,
                   view.rotation.D01,
                   view.rotation.D02,
                   view.rotation.D10,
                   view.rotation.D11,
                   view.rotation.D12,
                   view.rotation.D20,
                   view.rotation.D21,
                   view.rotation.D22);
            printf("view.position = { %f, %f, %f };\n",
                   view.position.X,
                   view.position.Y,
                   view.position.Z);
        }
    }
}

void Engine::mouse_callback_impl(int button, int action, int mods)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT && button != GLFW_MOUSE_BUTTON_RIGHT)
        return;

    if (action == GLFW_RELEASE)
    {
        state.mouse[(uint64)button] = false;
    }
    else if (action == GLFW_PRESS)
    {
        state.mouse[(uint64)button] = true;
    }
}

void Engine::scroll_callback_impl(double xoffset, double yoffset)
{
    config.radius += float(yoffset) * (state.keys[GLFW_KEY_LEFT_SHIFT] ? 10.f : 1.f);
    config.radius = std::max(config.radius, 0.f);
}

// Main update function
void Engine::tick()
{
    PROFILE_FUNCTION();

    frameIndex++;

    if (printMemoryStats)
    {
        clear_console();
        std::cout << Memory::get_stats_string();
    }

    videoManager.tick(*this);

    // Process different aspects of the engine in a structured way
    handleCameraAndReplayControls();
    processVoxelData();
    processEditing();
    updateTimingsAndStats();

    HACK_PROFILE_FRAME_ADVANCE();
}

/**
 * Handles camera movement and replay controls.
 * Processes keyboard inputs for camera positioning, manages camera transitions,
 * and handles replay recording/playback logic.
 */
void Engine::handleCameraAndReplayControls()
{
    // --- Camera Controls ---
    if (replayReader.is_empty())
    {
        handleCameraPresets();
        handleCameraMovement();
    }

    // Smoothly interpolate camera to the target view.
    if (moveToTarget)
    {
        targetLerpTime = clamp(targetLerpTime + targetLerpSpeed * dt, 0., 1.);
        view.position = lerp(initialView.position, targetView.position, targetLerpTime);
        view.rotation = Matrix3x3::FromQuaternion(Quaternion::Slerp(Matrix3x3::ToQuaternion(initialView.rotation), Matrix3x3::ToQuaternion(targetView.rotation), targetLerpTime));
    }

    // --- Replay Management ---
    if (replayReader.is_empty())
    {
        // Record current camera state
        replayWriter.add_action<ReplayActionSetLocation>(view.position);
        replayWriter.add_action<ReplayActionSetRotation>(view.rotation);
    }
    else if (!replayReader.at_end())
    {
        // Playback recorded actions
        replayReader.replay_frame();
        if (replayReader.at_end())
        {
            handleReplayCompletion();
        }
    }
}

/**
 * Processes keyboard inputs for predefined camera view presets.
 * Allows quick camera position changes with numpad keys.
 */
void Engine::handleCameraPresets()
{
    if (state.keys[GLFW_KEY_KP_0])
    {
        targetView.rotation = {-0.573465, 0.000000, -0.819230, -0.034067, 0.999135, 0.023847, 0.818522, 0.041585, -0.572969};
        targetView.position = {-13076.174715, -1671.669438, 5849.331627};
        init_target_lerp();
    }
    if (state.keys[GLFW_KEY_KP_1])
    {
        targetView.rotation = {0.615306, -0.000000, -0.788288, -0.022851, 0.999580, -0.017837, 0.787957, 0.028989, 0.615048};
        targetView.position = {-7736.138941, -2552.420373, -5340.566371};
        init_target_lerp();
    }
    if (state.keys[GLFW_KEY_KP_2])
    {
        targetView.rotation = {-0.236573, -0.000000, -0.971614, 0.025623, 0.999652, -0.006239, 0.971276, -0.026372, -0.236491};
        targetView.position = {-2954.821641, 191.883613, 4200.793442};
        init_target_lerp();
    }
    if (state.keys[GLFW_KEY_KP_3])
    {
        targetView.rotation = {0.590287, -0.000000, -0.807193, 0.150128, 0.982552, 0.109786, 0.793109, -0.185987, 0.579988};
        targetView.position = {-7036.452685, -3990.109906, 7964.129876};
        init_target_lerp();
    }
    if (state.keys[GLFW_KEY_KP_4])
    {
        targetView.rotation = {-0.222343, -0.000000, -0.974968, 0.070352, 0.997393, -0.016044, 0.972427, -0.072159, -0.221764};
        targetView.position = {762.379376, -935.456405, -358.642203};
        init_target_lerp();
    }
    if (state.keys[GLFW_KEY_KP_5])
    {
        targetView.rotation = {0, 0, -1, 0, 1, 0, 1, 0, 0};
        targetView.position = {-951.243605, 667.199855, -27.706481};
        init_target_lerp();
    }
    if (state.keys[GLFW_KEY_KP_6])
    {
        targetView.rotation = {0.095015, -0.000000, -0.995476, 0.130796, 0.991331, 0.012484, 0.986846, -0.131390, 0.094192};
        targetView.position = {652.972238, 73.188250, -209.028828};
        init_target_lerp();
    }
    if (state.keys[GLFW_KEY_KP_7])
    {
        targetView.rotation = {-0.004716, -0.000000, -0.999989, 0.583523, 0.812093, -0.002752, 0.812084, -0.583529, -0.003830};
        targetView.position = {-1261.247484, 1834.904220, -11.976059};
        init_target_lerp();
    }
    if (state.keys[GLFW_KEY_KP_8])
    {
        targetView.rotation = {0.019229, -0.000000, 0.999815, -0.040020, 0.999198, 0.000770, -0.999014, -0.040027, 0.019213};
        targetView.position = {-8998.476232, -2530.419704, -4905.593975};
        init_target_lerp();
    }

    if (state.keys[GLFW_KEY_KP_ADD])
        config.radius++;
}

/**
 * Handles keyboard inputs for manual camera movement.
 * Processes WASD for position, arrow keys / E/Q for rotation.
 */
void Engine::handleCameraMovement()
{
    double speed = length(make_double3(dagInfo.boundsAABBMax - dagInfo.boundsAABBMin)) / 100 * dt;
    double rotationSpeed = 2 * dt;

    if (state.keys[GLFW_KEY_LEFT_SHIFT])
        speed *= 10;

    // Position movement
    if (state.keys[GLFW_KEY_W])
    {
        view.position += speed * view.forward();
        moveToTarget = false;
    }
    if (state.keys[GLFW_KEY_S])
    {
        view.position -= speed * view.forward();
        moveToTarget = false;
    }
    if (state.keys[GLFW_KEY_D])
    {
        view.position += speed * view.right();
        moveToTarget = false;
    }
    if (state.keys[GLFW_KEY_A])
    {
        view.position -= speed * view.right();
        moveToTarget = false;
    }
    if (state.keys[GLFW_KEY_SPACE])
    {
        view.position += speed * view.up();
        moveToTarget = false;
    }
    if (state.keys[GLFW_KEY_LEFT_CONTROL])
    {
        view.position -= speed * view.up();
        moveToTarget = false;
    }

    // Rotation movement
    if (state.keys[GLFW_KEY_RIGHT] || state.keys[GLFW_KEY_E])
    {
        view.rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(rotationSpeed, Vector3::Up()));
        moveToTarget = false;
    }
    if (state.keys[GLFW_KEY_LEFT] || state.keys[GLFW_KEY_Q])
    {
        view.rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(-rotationSpeed, Vector3::Up()));
        moveToTarget = false;
    }
    if (state.keys[GLFW_KEY_DOWN])
    {
        view.rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(rotationSpeed, view.right()));
        moveToTarget = false;
    }
    if (state.keys[GLFW_KEY_UP])
    {
        view.rotation *= Matrix3x3::FromQuaternion(Quaternion::FromAngleAxis(-rotationSpeed, view.right()));
        moveToTarget = false;
    }
}

/**
 * Handles logic when replay playback is complete.
 * Manages benchmark statistics and restarts replay if needed.
 */
void Engine::handleReplayCompletion()
{
    if (firstReplay && REPLAY_TWICE)
    {
        printf("First replay ended, starting again now that everything is loaded in memory...\n");
        firstReplay = false;
        replayReader.reset_replay();
        statsRecorder.clear();
    }
    else
    {
#if BENCHMARK
        printf("Replay ended, saving stats... ");
        statsRecorder.write_csv();
#endif
#ifdef PROFILING_PATH
        hashDag.data.save_bucket_sizes(false);
#endif
        statsRecorder.clear();
        printf("Saved!\n");
    }
}

/**
 * Processes voxel data for paths, colors, and shadows.
 * Handles raycasting to determine which voxel is under the mouse,
 * resolves colors for visualization, and calculates shadows if enabled.
 */
void Engine::processVoxelData()
{
    // --- Resolve Paths ---
    // Determines which voxel is under the mouse
    double pathsTime = resolveVoxelPaths();
    statsRecorder.report("paths", pathsTime);

    // Track mouse position and get current path
    updateMousePath();

    // --- Resolve Colors ---
    // Update the colors for visualization
    double colorsTime = resolveVoxelColors();
    statsRecorder.report("colors", colorsTime);

    // --- Resolve Shadows ---
    // Calculate shadows if enabled
    double shadowsTime = resolveVoxelShadows();
    statsRecorder.report("shadows", shadowsTime);

    // Store timing information
    timings.pathsTime = pathsTime;
    timings.colorsTime = colorsTime;
    timings.shadowsTime = shadowsTime;
}

/**
 * Resolves paths to determine which voxel is under the mouse.
 * Uses raycasting against the current DAG representation.
 *
 * @return Time taken to resolve paths in milliseconds
 */
double Engine::resolveVoxelPaths()
{
    double pathsTime = 0;
    switch (config.currentDag)
    {
    case EDag::BasicDagUncompressedColors:
    case EDag::BasicDagCompressedColors:
    case EDag::BasicDagColorErrors:
        pathsTime = tracer->resolve_paths(view, dagInfo, basicDag);
        break;
    case EDag::HashDag:
        pathsTime = tracer->resolve_paths(view, dagInfo, hashDag);
        break;
    }
    return pathsTime;
}

/**
 * Updates the current mouse position and gets the corresponding voxel path.
 * Converts window mouse coordinates to image coordinates and updates the path.
 */
void Engine::updateMousePath()
{
    constexpr double xMultiplier = double(imageWidth) / windowWidth;
    constexpr double yMultiplier = double(imageHeight) / windowHeight;

    const uint32 posX = uint32(clamp<int32>(int32(xMultiplier * state.mousePosX), 0, imageWidth - 1));
    const uint32 posY = uint32(clamp<int32>(int32(yMultiplier * state.mousePosY), 0, imageHeight - 1));

    if (replayReader.is_empty())
    {
        config.path = tracer->get_path(posX, posY);
#if RECORD_TOOL_OVERLAY
        replayWriter.add_action<ReplayActionSetToolParameters>(config.path, config.copySourcePath, config.copyDestPath, config.radius, uint32(config.tool));
#endif
    }
}

/**
 * Resolves colors for visualization based on the current DAG and debug settings.
 *
 * @return Time taken to resolve colors in milliseconds
 */
double Engine::resolveVoxelColors()
{
    double colorsTime = 0;
    const uint32 debugColorsIndexLevel = basicDag.levels - 2 - config.debugColorsIndexLevel;
    const ToolInfo toolInfo{
        config.tool,
        config.path,
        config.radius,
        config.copySourcePath,
        config.copyDestPath};

    switch (config.currentDag)
    {
    case EDag::BasicDagUncompressedColors:
        colorsTime = tracer->resolve_colors(basicDag, basicDagUncompressedColors, config.debugColors,
                                            debugColorsIndexLevel, toolInfo);
        break;
    case EDag::BasicDagCompressedColors:
        colorsTime = tracer->resolve_colors(basicDag, basicDagCompressedColors, config.debugColors, debugColorsIndexLevel,
                                            toolInfo);
        break;
    case EDag::BasicDagColorErrors:
        colorsTime = tracer->resolve_colors(basicDag, basicDagColorErrors, config.debugColors,
                                            debugColorsIndexLevel, toolInfo);
        break;
    case EDag::HashDag:
        colorsTime = tracer->resolve_colors(hashDag, hashDagColors, config.debugColors, debugColorsIndexLevel, toolInfo);
        break;
    }

    return colorsTime;
}

/**
 * Resolves shadows and fog effects if enabled.
 *
 * @return Time taken to resolve shadows in milliseconds
 */
double Engine::resolveVoxelShadows()
{
    double shadowsTime = 0;
    if (shadows && ENABLE_SHADOWS)
    {
        switch (config.currentDag)
        {
        case EDag::BasicDagUncompressedColors:
        case EDag::BasicDagCompressedColors:
        case EDag::BasicDagColorErrors:
            shadowsTime = tracer->resolve_shadows(view, dagInfo, basicDag, shadowBias, fogDensity);
            break;
        case EDag::HashDag:
            shadowsTime = tracer->resolve_shadows(view, dagInfo, hashDag, shadowBias, fogDensity);
            break;
        }
    }
    return shadowsTime;
}

/**
 * Processes editing operations on the voxel data.
 * Handles different editing tools (Sphere, Cube, Copy, Fill, etc.)
 * when mouse buttons are pressed.
 */
void Engine::processEditing()
{
    // Only edit with HashDAG and not during replay
    if (config.currentDag == EDag::HashDag && replayReader.is_empty())
    {
        if (state.mouse[GLFW_MOUSE_BUTTON_LEFT] || state.mouse[GLFW_MOUSE_BUTTON_RIGHT])
        {
            // Handle copy source/destination selection for CubeCopy tool
            if (config.tool == ETool::CubeCopy && state.mouse[GLFW_MOUSE_BUTTON_RIGHT])
            {
                if (state.keys[GLFW_KEY_LEFT_SHIFT])
                {
                    config.copySourcePath = config.path;
                }
                else
                {
                    config.copyDestPath = config.path;
                }
            }

            // Apply the appropriate editing tool
            applyEditingTool();
        }
    }
}

/**
 * Applies the selected editing tool to modify the voxel data.
 * Different tools have different behaviors (add/remove, paint, noise, etc.).
 */
void Engine::applyEditingTool()
{
    const bool isAdding = state.mouse[GLFW_MOUSE_BUTTON_RIGHT];
    const float3 position = make_float3(config.path);

    // Apply different editors based on the selected tool
    switch (config.tool)
    {
    case ETool::Sphere:
        if (isAdding)
        {
            edit<SphereEditor<true>>(position, config.radius);
        }
        else
        {
            edit<SphereEditor<false>>(position, config.radius);
        }
        replayWriter.add_action<ReplayActionSphere>(position, config.radius, isAdding);
        break;

    case ETool::SpherePaint:
        edit<SpherePaintEditor>(position, config.radius);
        replayWriter.add_action<ReplayActionPaint>(position, config.radius);
        break;

    case ETool::SphereNoise:
        edit<SphereNoiseEditor>(hashDag, position, config.radius, isAdding);
        break;

    case ETool::Cube:
        if (isAdding)
        {
            edit<BoxEditor<true>>(position, config.radius);
        }
        else
        {
            edit<BoxEditor<false>>(position, config.radius);
        }
        replayWriter.add_action<ReplayActionCube>(position, config.radius, isAdding);
        break;

    case ETool::CubeCopy:
        if (!isAdding && config.radius >= 1)
        {
            const float3 src = make_float3(config.copySourcePath);
            const float3 dest = make_float3(config.copyDestPath);
            const Matrix3x3 transform = Matrix3x3::FromQuaternion(Quaternion::FromEuler(transformRotation / 180 * M_PI)) * transformScale;
            edit<CopyEditor>(hashDag, hashDagColors, src, dest, position, config.radius, transform, statsRecorder, enableSwirl, swirlPeriod);
            replayWriter.add_action<ReplayActionCopy>(position, src, dest, config.radius, transform, enableSwirl, swirlPeriod);
        }
        break;

    case ETool::CubeFill:
        const float3 center = position + (isAdding ? -1.f : 1.f) * round(2.f * make_float3(view.forward()));
        edit<FillEditorColors>(hashDag, hashDagColors, center, config.radius);
        replayWriter.add_action<ReplayActionFill>(center, config.radius);
        break;
    }
}

/**
 * Updates timing statistics and other metrics.
 * Records frame times, calculates delta time, and manages replay actions.
 */
void Engine::updateTimingsAndStats()
{
    auto currentTime = Utils::seconds();

    // Update timing information
    timings.totalTime = (currentTime - time) * 1e3;
    dt = currentTime - time;
    time = currentTime;

    // Record end of frame for replay or collect stats
    if (replayReader.is_empty())
    {
        replayWriter.add_action<ReplayActionEndFrame>();
    }
    else
    {
        recordReplayStats();
    }
}

/**
 * Records statistics during replay for benchmarking and analysis.
 */
void Engine::recordReplayStats()
{
    if (hashDag.data.is_valid())
    {
        statsRecorder.report("virtual_size", hashDag.data.get_virtual_used_size(false));
        statsRecorder.report("allocated_size", hashDag.data.get_allocated_pages_size());
        statsRecorder.report("color_size", hashDagColors.get_total_used_memory());
        statsRecorder.report("color_size undo_redo", undoRedo.get_total_used_memory());
#if SIMULATE_GC
        hashDag.simulate_remove_stale_nodes(statsRecorder);
#endif
    }
    statsRecorder.next_frame();
}

//-----------------------------------------------------------------------------
// Initialization and Cleanup
//-----------------------------------------------------------------------------
void Engine::init(bool inheadLess)
{
    PROFILE_FUNCTION();

    headLess = inheadLess;

    if (!headLess)
    {
        init_graphics();
    }

    tracer = std::make_unique<DAGTracer>(headLess);
    image = tracer->get_colors_image();
    time = Utils::seconds();
}

void Engine::init_graphics()
{
    PROFILE_FUNCTION();

    initializeGLFW();
    initializeGLEW();
    setupOpenGLState();
    loadFonts();
    createFullScreenQuad();
    setupMarchingCubes();
}

/**
 * Initializes GLFW, creates a window, and sets up callbacks.
 * Handles window creation and input callback registration.
 */
void Engine::initializeGLFW()
{
    // Initialize GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(1);
    }

    // Set GLFW window hints
    glfwWindowHint(GLFW_SAMPLES, 4);                               // Anti-aliasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);                 // OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);                 // OpenGL 3.3
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // Mac compatibility
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Core profile
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);                 // Prevent minimizing on focus loss

    // Create a window and its OpenGL context
    window = glfwCreateWindow(windowWidth, windowHeight, "DAG Edits", NULL, NULL);
    if (window == NULL)
    {
        fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible.\n");
        glfwTerminate();
        exit(1);
    }

    // Set up window and callbacks
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSwapInterval(0); // Disable v-sync

    // Ensure we can capture the escape key
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
}

/**
 * Initializes GLEW to access OpenGL extensions.
 * Must be called after GLFW is initialized and a context is created.
 */
void Engine::initializeGLEW()
{
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(1);
    }
}

/**
 * Sets up the initial OpenGL state.
 * Configures background color, depth testing, and other rendering settings.
 */
void Engine::setupOpenGLState()
{
    // Set dark blue background color
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS); // Accept fragment if closer to camera than former fragment
}

/**
 * Loads and initializes fonts for UI rendering.
 */
void Engine::loadFonts()
{
    // Load font files and create rendering context
    fontctx = glf::make_context(
        "assets/droid-sans-mono/DroidSansMonoDotted.ttf",
        windowWidth, windowHeight,
        "assets/noto-emoji/NotoEmoji-Regular.ttf");

    // Create text buffers for static and dynamic text
    dynamicText = glf::make_buffer();
    staticText = glf::make_buffer();
}

/**
 * Creates and sets up a full-screen quad for rendering the DAG visualization.
 * Sets up vertex buffers, attributes, and shaders.
 */
void Engine::createFullScreenQuad()
{
    // Create a vertex array object for the quad
    glGenVertexArrays(1, &fsvao);
    glBindVertexArray(fsvao);

    // Load and compile shaders
    programID = LoadShaders("src/shaders/TransformVertexShader.glsl", "src/shaders/TextureFragmentShader.glsl");
    textureID = glGetUniformLocation(programID, "myTextureSampler");

    // Define quad vertices (two triangles forming a rectangle)
    static const GLfloat g_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f, // Bottom left
        1.0f, -1.0f, 0.0f,  // Bottom right
        1.0f, 1.0f, 0.0f,   // Top right
        -1.0f, -1.0f, 0.0f, // Bottom left
        1.0f, 1.0f, 0.0f,   // Top right
        -1.0f, 1.0f, 0.0f   // Top left
    };

    // Define UV coordinates for texture mapping
    static const GLfloat g_uv_buffer_data[] = {
        0.0f, 1.0f, // Bottom left
        1.0f, 1.0f, // Bottom right
        1.0f, 0.0f, // Top right
        0.0f, 1.0f, // Bottom left
        1.0f, 0.0f, // Top right
        0.0f, 0.0f  // Top left
    };

    // Create and populate vertex buffer
    GLuint vertexBuffer = 0;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

    // Create and populate UV buffer
    GLuint uvBuffer = 0;
    glGenBuffers(1, &uvBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);

    // Set up vertex attributes
    glEnableVertexAttribArray(0); // Position attribute
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glVertexAttribPointer(
        0,        // Attribute index
        3,        // Size (x,y,z)
        GL_FLOAT, // Type
        GL_FALSE, // Normalized?
        0,        // Stride
        (void *)0 // Array buffer offset
    );

    // Set up UV attributes
    glEnableVertexAttribArray(1); // UV attribute
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
    glVertexAttribPointer(
        1,        // Attribute index
        2,        // Size (u,v)
        GL_FLOAT, // Type
        GL_FALSE, // Normalized?
        0,        // Stride
        (void *)0 // Array buffer offset
    );

    // Clean up
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &uvBuffer);
}

//-----------------------------------------------------------------------------
// Main Loop and Rendering
//-----------------------------------------------------------------------------
void Engine::loop()
{
    if (headLess)
    {
        loop_headless();
    }
    else
    {
        loop_graphics();
    }
}

void Engine::loop_headless()
{
    PROFILE_FUNCTION();

    while (!replayReader.at_end())
    {
        MARK_FRAME();
        tick();
    }
}

void Engine::loop_graphics()
{
    PROFILE_FUNCTION();

    do
    {
        MARK_FRAME();

        // Process a single frame (input, simulation, rendering)

        // Update inputs and poll window events
        pollInputs();

        // Update simulation state
        tick();

        // Render the scene and UI
        renderFrame();

    } // Continue looping until the escape key is pressed or the window is closed
    while (!shouldExitApplication());
}

//-----------------------------------------------------------------------------
// Scene Rendering
//-----------------------------------------------------------------------------
/**
 * Renders a complete frame including the main scene and UI.
 * Manages OpenGL state for rendering different components.
 */
void Engine::renderFrame()
{
    // Render the main scene (DAG visualization)
    renderMainScene();

    // Render UI elements if enabled
    if (showUI)
    {
        renderUI();
    }

    // Swap buffers to display the rendered frame
    glfwSwapBuffers(window);
}

//-----------------------------------------------------------------------------
// User Interface and Input
//-----------------------------------------------------------------------------
/**
 * Polls window inputs and events for the current frame.
 * Updates mouse position and handles GLFW events.
 */
void Engine::pollInputs()
{
    // Get current mouse cursor position
    glfwGetCursorPos(window, &state.mousePosX, &state.mousePosY);

    // Process any pending events
    glfwPollEvents();

    // Update window title
    glfwSetWindowTitle(window, "HashDag");
}

/**
 * Renders the main scene (DAG visualization).
 * Sets up shaders, binds textures, and draws the fullscreen quad.
 */
void Engine::renderMainScene()
{
    // Clear the color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (renderMode == RenderMode::MarchingCubes)
    {
        // Enable depth testing for 3D rendering
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS); // Accept fragment if closer to camera than former fragment

        // Render marching cubes instead of a simple cube
        renderMarchingCubes();

        // Disable depth testing when done
        glDisable(GL_DEPTH_TEST);
    }
    else
    {
        // Use our shader program for default rendering
        glUseProgram(programID);

        // Bind the texture containing the DAG visualization
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, image);
        glUniform1i(textureID, 0);

        // Draw the fullscreen quad
        glBindVertexArray(fsvao);
        glDrawArrays(GL_TRIANGLES, 0, 6); // 6 vertices for 2 triangles

        // Clean up
        glBindVertexArray(0);
        glUseProgram(0);
    }
}

// Helper function for dot product calculation
inline float dot(const Vector3 &a, const Vector3 &b)
{
    return (float)(a.X * b.X + a.Y * b.Y + a.Z * b.Z);
}

//-----------------------------------------------------------------------------
// UI Components and Rendering
//-----------------------------------------------------------------------------
/**
 * Renders UI elements including statistics, tool info, and debug data.
 * Manages text rendering and OpenGL state for 2D rendering.
 */
void Engine::renderUI()
{
    // Configure OpenGL for 2D text rendering
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Format helpers for consistent text formatting
    renderUIElements();

    // Draw text buffers
    glf::draw_buffer(fontctx, staticText);
    glf::draw_buffer(fontctx, dynamicText);
    glf::clear_buffer(dynamicText);

    // Restore OpenGL state for 3D rendering
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

/**
 * Renders all UI elements including stats, memory info, and tool settings.
 * Creates and formats text for display.
 */
void Engine::renderUIElements()
{
    // Use formatting types from the text rendering library
    using glf::EFmt;

    // Formatter functions for consistent text formatting
    auto const time_ = [&](auto &&T)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(2) << T << "ms";
        return oss.str();
    };

    auto const mb_ = [&](auto &&T)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(1) << T << "MB";
        return oss.str();
    };

    auto const cmb_ = [&](auto &&T, auto &&U)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(1) << T << "";
        oss << " (" << std::fixed << std::setw(6) << std::setprecision(1) << U << "MB)";
        return oss.str();
    };

    auto const mbx_ = [&](auto &&T, auto &&U)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(1) << T << "MB";
        oss << " (+" << std::fixed << std::setw(6) << std::setprecision(1) << U << "MB)";
        return oss.str();
    };

    auto const mb2_ = [&](auto &&T, auto &&U)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(1) << T << "MB";
        oss << " / " << std::fixed << std::setw(6) << std::setprecision(1) << U << "MB";
        return oss.str();
    };

    auto const vector3_ = [&](const Vector3 &V)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << V.X << ", " << V.Y << ", " << V.Z;
        return oss.str();
    };

    auto const count_ = [&](auto &&T)
    {
        std::ostringstream oss;
        oss << std::scientific << std::setw(4) << std::setprecision(3) << T;
        return oss.str();
    };

    // Draw all UI elements
    auto const draw = [&](auto &&F)
    {
        float y = windowHeight - 42.f;
        float const hx = 75.f;
        float const sx = 120.f;
        float const dx = 305.f;

// DAG and tool info section
#define STRINGIFY0_(x) #x
#define STRINGIFY_(x) STRINGIFY0_(x)
        static constexpr char scene[] = "Scene " STRINGIFY_(SCENE) " (2^" STRINGIFY_(SCENE_DEPTH) ") using ";
        F(hx, y, EFmt::glow, scene, dx + sizeof(scene) * 6, dag_to_string(config.currentDag));
        y -= 24.f;
#undef STRINGIFY_
#undef STRINGIFY0_

        F(hx, y, EFmt::glow, "Active tool:", dx, tool_to_string(config.tool));
        y -= 24.f;

        // Tool-specific settings
        renderToolSpecificUI(F, hx, dx, y);
        y -= 32.f;

        // Timing statistics
        renderTimingStatistics(F, hx, sx, dx, y);
        y -= 24.f;

        // Memory statistics
        renderMemoryStatistics(F, hx, sx, dx, y);
        y -= 24.f;

        // Edit statistics
        renderEditStatistics(F, hx, sx, dx, y);
    };

    // Apply the draw function to add lines to the dynamic text buffer
    draw([&](float aX1, float aY, EFmt aFmt, auto &&aTxt1, float aX2, auto &&aTxt2)
         {
        if (aX1 > 0.0f)
            glf::add_line(fontctx, dynamicText, aFmt, aX1, aY, aTxt1);
        if (aX2 > 0.0f)
            glf::add_line(fontctx, dynamicText, aFmt, aX2, aY, aTxt2); });
}

/**
 * Renders tool-specific UI elements based on the current selected tool.
 */
template <typename FormatterFunc>
void Engine::renderToolSpecificUI(FormatterFunc &F, float hx, float dx, float &y)
{
    using glf::EFmt;

    auto const vector3_ = [&](const Vector3 &V)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << V.X << ", " << V.Y << ", " << V.Z;
        return oss.str();
    };

    if (config.tool == ETool::CubeCopy)
    {
#if COPY_APPLY_TRANSFORM
        F(hx, y, EFmt::glow, "Rotation:", dx, vector3_(transformRotation));
        y -= 24.f;
        F(hx, y, EFmt::glow, "Scale:", dx, std::to_string(transformScale));
        y -= 24.f;
#endif
#if COPY_CAN_APPLY_SWIRL
        F(hx, y, EFmt::glow, "Swirl:", dx, enableSwirl ? "ON" : "OFF");
        y -= 24.f;
        F(hx, y, EFmt::glow, "Swirl period:", dx, std::to_string(swirlPeriod));
        y -= 24.f;
#endif
    }
}

/**
 * Renders timing statistics for the current frame.
 */
template <typename FormatterFunc>
void Engine::renderTimingStatistics(FormatterFunc &F, float hx, float sx, float dx, float &y)
{
    using glf::EFmt;

    auto const time_ = [&](auto &&T)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(2) << T << "ms";
        return oss.str();
    };

    const double editingAndUploading =
        lastEditFrame == frameIndex ? statsRecorder.get_value_in_frame(lastEditTimestamp, "total edits") +
                                          statsRecorder.get_value_in_frame(lastEditTimestamp, "upload_to_gpu") +
                                          statsRecorder.get_value_in_frame(lastEditTimestamp, "creating edit tool")
                                    : 0;

    F(hx, y, EFmt::large, "Timings", -1.f, nullptr);
    y -= 32.f;
    F(sx, y, EFmt::glow, "Trace primary", dx, time_(timings.pathsTime));
    y -= 24.f;
    F(sx, y, EFmt::glow, "Trace shadow", dx, time_(timings.shadowsTime));
    y -= 24.f;
    F(sx, y, EFmt::glow, "Resolve colors", dx, time_(timings.colorsTime));
    y -= 24.f;
    F(sx, y, EFmt::glow, "Edit & Upload", dx, time_(editingAndUploading));
    y -= 24.f;
    F(sx, y, EFmt::glow, "Total", dx, time_(timings.totalTime));
    y -= 24.f;
}

/**
 * Renders memory usage statistics.
 */
template <typename FormatterFunc>
void Engine::renderMemoryStatistics(FormatterFunc &F, float hx, float sx, float dx, float &y)
{
    using glf::EFmt;

    auto const mb_ = [&](auto &&T)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(1) << T << "MB";
        return oss.str();
    };

    auto const cmb_ = [&](auto &&T, auto &&U)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(1) << T << "";
        oss << " (" << std::fixed << std::setw(6) << std::setprecision(1) << U << "MB)";
        return oss.str();
    };

    auto const mbx_ = [&](auto &&T, auto &&U)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(1) << T << "MB";
        oss << " (+" << std::fixed << std::setw(6) << std::setprecision(1) << U << "MB)";
        return oss.str();
    };

    auto const mb2_ = [&](auto &&T, auto &&U)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setw(6) << std::setprecision(1) << T << "MB";
        oss << " / " << std::fixed << std::setw(6) << std::setprecision(1) << U << "MB";
        return oss.str();
    };

    F(hx, y, EFmt::large, "Memory", -1.f, nullptr);
    y -= 32.f;
    F(sx, y, EFmt::glow, "Page pool", dx, cmb_(hashDag.data.get_total_pages(), hashDag.data.get_pool_size()));
    y -= 24.f;
    F(sx, y, EFmt::glow, "  used", dx, cmb_(hashDag.data.get_allocated_pages(), hashDag.data.get_allocated_pages_size()));
    y -= 24.f;
    F(sx, y, EFmt::glow, "Page table", dx, mb_(hashDag.data.get_page_table_size()));
    y -= 24.f;

#if USE_VIDEO
    y -= 32.f - 24.f;
    F(hx, y, EFmt::glow, "  Total (GPU/CPU)", dx, mb2_(Utils::to_MB(Memory::get_gpu_allocated_memory()), Utils::to_MB(Memory::get_cpu_allocated_memory() + Memory::get_cxx_cpu_allocated_memory())));
    y -= 24.f;
#else  // !USE_VIDEO
    F(sx, y, EFmt::glow, "Total GPU", dx, mb_(Utils::to_MB(Memory::get_gpu_allocated_memory())));
    y -= 24.f;
    F(sx, y, EFmt::glow, "Total CPU", dx, mbx_(Utils::to_MB(Memory::get_cpu_allocated_memory()), Utils::to_MB(Memory::get_cxx_cpu_allocated_memory())));
    y -= 24.f;
#endif // ~ USE_VIDEO
}

/**
 * Renders edit statistics like voxel counts and node counts.
 */
template <typename FormatterFunc>
void Engine::renderEditStatistics(FormatterFunc &F, float hx, float sx, float dx, float &y)
{
    using glf::EFmt;

    auto const count_ = [&](auto &&T)
    {
        std::ostringstream oss;
        oss << std::scientific << std::setw(4) << std::setprecision(3) << T;
        return oss.str();
    };

    F(hx, y, EFmt::large, "Edits", -1.f, nullptr);
    y -= 32.f;
    F(sx, y, EFmt::glow, "Num Voxels", dx, count_(statsRecorder.get_value_in_frame(lastEditTimestamp, "num voxels")));
    y -= 24.f;
    F(sx, y, EFmt::glow, "Num Nodes", dx, count_(statsRecorder.get_value_in_frame(lastEditTimestamp, "num nodes")));
    y -= 24.f;

    // Add camera information section
    F(hx, y, EFmt::large, "Camera", -1.f, nullptr);
    y -= 32.f;

    // Format position with fixed precision
    std::ostringstream posStream;
    posStream << std::fixed << std::setprecision(2)
              << "X=" << view.position.X
              << ", Y=" << view.position.Y
              << ", Z=" << view.position.Z;
    F(sx, y, EFmt::glow, "Position", dx, posStream.str());
    y -= 24.f;

    // For rotation, show a simplified version (just the forward vector)
    Vector3 forward = view.forward();
    std::ostringstream dirStream;
    dirStream << std::fixed << std::setprecision(2)
              << "X=" << forward.X
              << ", Y=" << forward.Y
              << ", Z=" << forward.Z;
    F(sx, y, EFmt::glow, "Direction", dx, dirStream.str());
    y -= 24.f;
}

/**
 * Checks if the application should exit based on user input or window state.
 *
 * @return True if the application should exit, false otherwise
 */
bool Engine::shouldExitApplication()
{
    bool escapePressed = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    bool windowClosed = glfwWindowShouldClose(window) != 0;

#ifdef EXIT_AFTER_REPLAY
    bool replayEnded = replayReader.at_end();
    return escapePressed || windowClosed || replayEnded;
#else
    return escapePressed || windowClosed;
#endif
}

//-----------------------------------------------------------------------------
// Cleanup and Resource Management
//-----------------------------------------------------------------------------
/**
 * Cleans up resources used by the engine.
 * Frees all memory and destroys OpenGL resources.
 */
void Engine::destroy()
{
    glf::destroy_buffer(staticText);
    glf::destroy_buffer(dynamicText);
    glf::destroy_context(fontctx);

    glDeleteVertexArrays(1, &fsvao);

    tracer.reset();
    basicDag.free();
    basicDagCompressedColors.free();
    basicDagUncompressedColors.free();
    basicDagColorErrors.free();
    hashDag.free();
    hashDagColors.free();
    undoRedo.free();

    cleanupMarchingCubes(); // Clean up marching cubes resources
}

void Engine::toggle_fullscreen()
{
    if (!fullscreen)
    {
        fullscreen = true;
        GLFWmonitor *primary = glfwGetPrimaryMonitor();
        const GLFWvidmode *mode = glfwGetVideoMode(primary);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, mode->width, mode->height, mode->refreshRate);
    }
    else
    {
        fullscreen = false;
        glfwSetWindowMonitor(window, NULL, 0, 0, windowWidth, windowHeight, -1);
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
}

/**
 * Sets up the marching cubes renderer.
 * Creates shaders, compiles them, and sets up initial state.
 */
void Engine::setupMarchingCubes()
{
    // Create vertex shader - pass position to fragment shader
    const char *vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;        // Vertex position
        layout (location = 1) in vec3 aNormal;     // Vertex normal

        uniform mat4 model;      // Model transformation matrix
        uniform mat4 view;       // View matrix
        uniform mat4 projection; // Projection matrix

        out vec3 fragPosition;   // Fragment position in world space
        out vec3 fragNormal;     // Fragment normal
        out vec3 worldPosition;  // World position for coloring

        void main() {
            // Transform vertex position to world space
            fragPosition = vec3(model * vec4(aPos, 1.0));
            // Transform normal to world space
            fragNormal = mat3(transpose(inverse(model))) * aNormal;
            // Save world position for coloring
            worldPosition = fragPosition;
            
            // Calculate final vertex position
            gl_Position = projection * view * model * vec4(aPos, 1.0);
        }
    )";

    // Create fragment shader - use world position for coloring
    const char *fragmentShaderSource = R"(
        #version 330 core
        out vec4 fragColor;      // Final fragment color output

        in vec3 fragPosition;    // Fragment position in world space
        in vec3 fragNormal;      // Fragment normal
        in vec3 worldPosition;   // World position for coloring

        void main() {
            // Normalize world position to [0,1] range for coloring
            vec3 normalizedPos = normalize(worldPosition) * 0.5 + 0.5;
            
            // Use world position for coloring (xyz maps to rgb)
            vec3 positionColor = normalizedPos;
            
            // Apply some normal influence to keep surface details visible
            vec3 normalColor = normalize(fragNormal) * 0.5 + 0.5;
            
            // Blend position-based and normal-based coloring
            vec3 finalColor = positionColor * 0.7 + normalColor * 0.3;
            
            // Output final color
            fragColor = vec4(finalColor, 1.0);
        }
    )";

    // Compile shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Check for compilation errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n%s\n", infoLog);
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Check for compilation errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n%s\n", infoLog);
    }

    // Create shader program
    marchingCubesShaderProgram = glCreateProgram();
    glAttachShader(marchingCubesShaderProgram, vertexShader);
    glAttachShader(marchingCubesShaderProgram, fragmentShader);
    glLinkProgram(marchingCubesShaderProgram);

    // Check for linking errors
    glGetProgramiv(marchingCubesShaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(marchingCubesShaderProgram, 512, NULL, infoLog);
        printf("ERROR::SHADER::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
    }

    // Get uniform locations
    mcModelLoc = glGetUniformLocation(marchingCubesShaderProgram, "model");
    mcViewLoc = glGetUniformLocation(marchingCubesShaderProgram, "view");
    mcProjLoc = glGetUniformLocation(marchingCubesShaderProgram, "projection");

    // Clean up shaders after linking
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Generate initial geometry
    updateMarchingCubes();
}

/**
 * Updates the marching cubes mesh based on the current parameters.
 * Generates a new density field and processes it using marching cubes.
 */
void Engine::updateMarchingCubes()
{
    // Generate density field from the current DAG
    std::vector<float> densityField;

    densityField = generateHashDAGField(m_gridSize, 0.0f);

    // Generate marching cubes triangles (vertices and normals)
    std::vector<float> vertexData = marchingCubes(densityField, m_gridSize, 0.0f); // -1.0 to 1.0 range, 0.0f isosurface, higher value results in seeing less holes

    // Update vertex count
    marchingCubesVertexCount = static_cast<GLuint>(vertexData.size() / 6); // 6 floats per vertex (pos + normal)

    std::cout << "Generated " << marchingCubesVertexCount / 3 << " triangles for marching cubes." << std::endl;

    // Create VAO and VBO if not already created
    if (marchingCubesVAO == 0)
    {
        glGenVertexArrays(1, &marchingCubesVAO);
        glGenBuffers(1, &marchingCubesVBO);
    }

    // Bind VAO and update VBO
    glBindVertexArray(marchingCubesVAO);
    glBindBuffer(GL_ARRAY_BUFFER, marchingCubesVBO);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_STATIC_DRAW);

    // Set attribute pointers
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

/**
 * Renders the marching cubes mesh.
 * Sets up viewing matrices, lighting, and draws the mesh.
 */
void Engine::renderMarchingCubes()
{

    // For debugging, force update on every frame to catch any changes
    // updateMarchingCubes();

    // Update only when DAG changes
    static EDag lastDag = static_cast<EDag>(-1); // Use invalid value for first run
    if (lastDag != config.currentDag)
    {
        updateMarchingCubes();
        lastDag = config.currentDag;
    }

    // Check if we have any vertices to render
    if (marchingCubesVertexCount == 0)
    {
        // Nothing to render
        return;
    }

    // Use our shader program
    glUseProgram(marchingCubesShaderProgram);

    // Create view matrix from the engine's camera view
    Vector3 forward = view.forward();
    Vector3 right = view.right();
    Vector3 up = view.up();
    Vector3 pos = view.position;

    // Create a proper view matrix using the camera's orientation vectors
    float viewMatrix[16] = {
        (float)right.X, (float)up.X, (float)(-forward.X), 0.0f,
        (float)right.Y, (float)up.Y, (float)(-forward.Y), 0.0f,
        (float)right.Z, (float)up.Z, (float)(-forward.Z), 0.0f,
        (float)(-dot(right, pos)), (float)(-dot(up, pos)), (float)(dot(forward, pos)), 1.0f};

    // Create perspective projection matrix
    float aspect = (float)windowWidth / (float)windowHeight;
    float fov = 45.0f * ((float)M_PI / 180.0f);
    float near = 0.1f;    // Reduce near plane distance to see closer objects
    float far = 10000.0f; // Increase far plane to see distant objects

    float f = 1.0f / tanf(fov / 2.0f);
    float projMatrix[16] = {
        f / aspect, 0.0f, 0.0f, 0.0f,
        0.0f, f, 0.0f, 0.0f,
        0.0f, 0.0f, (far + near) / (near - far), -1.0f,
        0.0f, 0.0f, (2.0f * far * near) / (near - far), 0.0f};

    // Scale and position for the marching cubes mesh
    float scaleFactor = 10.0f;
    float offset = m_gridSize / 2.0f;
    float posX = -offset * scaleFactor;
    float posY = -offset * scaleFactor;
    float posZ = -offset * scaleFactor;

    // Position based on camera view or tool position
    // float posX = (float)view.position.X + -offset * scaleFactor;
    // float posY = -offset * scaleFactor;
    // float posZ = (float)view.position.Z + -offset * scaleFactor;
    // float posX = (float)config.path.x - offset * scaleFactor;
    // float posY = (float)config.path.y - offset * scaleFactor;
    // float posZ = (float)config.path.z - offset * scaleFactor;

    // Create model matrix with scale and position
    float modelMatrix[16] = {
        scaleFactor, 0.0f, 0.0f, 0.0f,
        0.0f, scaleFactor, 0.0f, 0.0f,
        0.0f, 0.0f, scaleFactor, 0.0f,
        posX, posY, posZ, 1.0f};

    // Set uniforms
    glUniformMatrix4fv(mcModelLoc, 1, GL_FALSE, modelMatrix);
    glUniformMatrix4fv(mcViewLoc, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(mcProjLoc, 1, GL_FALSE, projMatrix);

    // Enable depth testing for proper rensdering
    glEnable(GL_DEPTH_TEST);

    // Render the mesh
    glBindVertexArray(marchingCubesVAO);
    glDrawArrays(GL_TRIANGLES, 0, marchingCubesVertexCount);

    glBindVertexArray(0);
}

/**
 * Cleans up resources used by the marching cubes renderer.
 */
void Engine::cleanupMarchingCubes()
{
    if (marchingCubesVAO != 0)
    {
        glDeleteVertexArrays(1, &marchingCubesVAO);
        marchingCubesVAO = 0;
    }

    if (marchingCubesVBO != 0)
    {
        glDeleteBuffers(1, &marchingCubesVBO);
        marchingCubesVBO = 0;
    }

    if (marchingCubesShaderProgram != 0)
    {
        glDeleteProgram(marchingCubesShaderProgram);
        marchingCubesShaderProgram = 0;
    }
}

/**
 * Implements the marching cubes algorithm.
 * Processes a density field to generate triangles representing an isosurface.
 *
 * @param field The density field
 * @param gridSize Size of the grid (number of cells per dimension)
 * @param isoLevel The value at which to extract the isosurface
 * @return A flat vector of vertex data (positions and normals)
 */
std::vector<float> Engine::marchingCubes(const std::vector<float> &field, int gridSize, float isoLevel)
{
    std::vector<float> vertexData;

    // Define the 8 corners of a cube
    const int cornerOffsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

    // Define the 12 edges of a cube (pairs of corner indices)
    const int edgeCorners[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

    // Helper function to get field value at grid position
    auto getFieldValue = [&](int x, int y, int z) -> float
    {
        if (x < 0 || y < 0 || z < 0 || x >= gridSize || y >= gridSize || z >= gridSize)
            return 0.0f;

        return field[x + y * gridSize + z * gridSize * gridSize];
    };

    // Helper function to interpolate vertex position
    auto vertexInterp = [&](float isoLevel, float v1, float v2, float x1, float y1, float z1, float x2, float y2, float z2)
    {
        if (std::abs(isoLevel - v1) < 0.00001f)
            return std::array<float, 3>{x1, y1, z1};
        if (std::abs(isoLevel - v2) < 0.00001f)
            return std::array<float, 3>{x2, y2, z2};
        if (std::abs(v1 - v2) < 0.00001f)
            return std::array<float, 3>{x1, y1, z1};

        float t = (isoLevel - v1) / (v2 - v1);
        return std::array<float, 3>{
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1),
            z1 + t * (z2 - z1)};
    };

    // Process each cube in the grid
    for (int z = 0; z < gridSize - 1; z++)
    {
        for (int y = 0; y < gridSize - 1; y++)
        {
            for (int x = 0; x < gridSize - 1; x++)
            {
                // Get the field values at the cube corners
                float cornerValues[8];
                for (int i = 0; i < 8; i++)
                {
                    int cx = x + cornerOffsets[i][0];
                    int cy = y + cornerOffsets[i][1];
                    int cz = z + cornerOffsets[i][2];
                    cornerValues[i] = getFieldValue(cx, cy, cz);
                }

                // Determine which vertices are inside the isosurface
                unsigned int cubeIndex = 0;
                for (int i = 0; i < 8; i++)
                {
                    if (cornerValues[i] < isoLevel)
                        cubeIndex |= (1 << i);
                }

                // If the cube is entirely inside or outside the isosurface, skip it
                if (MarchingCubesTables::edgeTable[cubeIndex] == 0)
                    continue;

                // Calculate intersection points for each edge
                std::array<std::array<float, 3>, 12> intersections;
                for (int i = 0; i < 12; i++)
                {
                    if (MarchingCubesTables::edgeTable[cubeIndex] & (1 << i))
                    {
                        int c1 = edgeCorners[i][0];
                        int c2 = edgeCorners[i][1];

                        float v1 = cornerValues[c1];
                        float v2 = cornerValues[c2];

                        float x1 = static_cast<float>(x + cornerOffsets[c1][0]);
                        float y1 = static_cast<float>(y + cornerOffsets[c1][1]);
                        float z1 = static_cast<float>(z + cornerOffsets[c1][2]);

                        float x2 = static_cast<float>(x + cornerOffsets[c2][0]);
                        float y2 = static_cast<float>(y + cornerOffsets[c2][1]);
                        float z2 = static_cast<float>(z + cornerOffsets[c2][2]);

                        intersections[i] = vertexInterp(isoLevel, v1, v2, x1, y1, z1, x2, y2, z2);
                    }
                }

                // Create triangles based on the triangulation table
                for (int i = 0; MarchingCubesTables::triTable[cubeIndex][i] != -1; i += 3)
                {
                    // Get the three vertices of the triangle
                    auto v1 = intersections[MarchingCubesTables::triTable[cubeIndex][i]];
                    auto v2 = intersections[MarchingCubesTables::triTable[cubeIndex][i + 1]];
                    auto v3 = intersections[MarchingCubesTables::triTable[cubeIndex][i + 2]];

                    // Calculate normal
                    float nx = (v2[1] - v1[1]) * (v3[2] - v1[2]) - (v2[2] - v1[2]) * (v3[1] - v1[1]);
                    float ny = (v2[2] - v1[2]) * (v3[0] - v1[0]) - (v2[0] - v1[0]) * (v3[2] - v1[2]);
                    float nz = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0]);

                    // Normalize
                    float len = sqrtf(nx * nx + ny * ny + nz * nz);
                    if (len > 0)
                    {
                        nx /= len;
                        ny /= len;
                        nz /= len;
                    }

                    // Add the first vertex
                    vertexData.push_back(v1[0]);
                    vertexData.push_back(v1[1]);
                    vertexData.push_back(v1[2]);
                    vertexData.push_back(nx);
                    vertexData.push_back(ny);
                    vertexData.push_back(nz);

                    // Add the second vertex
                    vertexData.push_back(v2[0]);
                    vertexData.push_back(v2[1]);
                    vertexData.push_back(v2[2]);
                    vertexData.push_back(nx);
                    vertexData.push_back(ny);
                    vertexData.push_back(nz);

                    // Add the third vertex
                    vertexData.push_back(v3[0]);
                    vertexData.push_back(v3[1]);
                    vertexData.push_back(v3[2]);
                    vertexData.push_back(nx);
                    vertexData.push_back(ny);
                    vertexData.push_back(nz);
                }
            }
        }
    }

    return vertexData;
}

/**
 * Samples the HashDAG to generate a density field for marching cubes.
 *
 * @param gridSize Size of the grid (number of cells per dimension)
 * @param scale Scale factor for the grid coordinates
 * @return A density field representing the DAG structure
 */
std::vector<float> Engine::generateHashDAGField(int gridSize, float scale)
{
    // very high memory usage
    std::vector<float> field(gridSize * gridSize * gridSize);

    // Get the current DAG
    if (!is_dag_valid(config.currentDag))
    {
        std::cout << "No valid DAG" << std::endl;
        return field; // Return empty field if no valid DAG
    }
    std::cout << "Valid DAG: " << dag_to_string(config.currentDag) << std::endl;

    // If tool position is non-zero, use that instead
    int worldCenterX = (config.path.x != 0) ? config.path.x : 0;
    int worldCenterY = (config.path.y != 0) ? config.path.y : 0;
    int worldCenterZ = (config.path.z != 0) ? config.path.z : 0;

    std::cout << "Using world center: (" << worldCenterX << "," << worldCenterY << "," << worldCenterZ << ")" << std::endl;

    int sampleRadius = gridSize / 2;
    std::cout << "Sampling area: center=(" << worldCenterX << "," << worldCenterY << "," << worldCenterZ
              << "), radius=" << sampleRadius << std::endl;

    // Sample the DAG to generate the field
    for (int z = 0; z < gridSize; z++)
    {
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                // Transform grid coordinates to world space, centered on worldCenter
                int worldX = worldCenterX - sampleRadius + static_cast<int>(x);
                int worldY = worldCenterY - sampleRadius + static_cast<int>(y);
                int worldZ = worldCenterZ - sampleRadius + static_cast<int>(z);

                // Create path from world coordinates
                Path path(worldX, worldY, worldZ);

                // Check if this voxel exists in the DAG
                bool voxelExists = false;

                // Only HashDag is supported with DAGUtils::get_value
                if (config.currentDag == EDag::HashDag)
                {
                    voxelExists = DAGUtils::get_value(hashDag, path);
                }

                // Set the field value (-1 = inside, +1 = outside)
                int index = x + y * gridSize + z * gridSize * gridSize;
                field[index] = voxelExists ? -1.0f : 1.0f;
            }
        }
    }

    return field;
}