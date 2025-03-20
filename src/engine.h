#pragma once

#include "typedefs.h"

#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "camera_view.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/basic_dag/basic_dag.h"
#include "dag_tracer.h"
#include "replay.h"
#include "video.h"

#include "glfont.h"

// Include necessary headers
#include <array>
#include <functional>

// Forward declarations
class MarchingCubes;

/// Available DAG types
enum class EDag
{
    BasicDagUncompressedColors,
    BasicDagCompressedColors,
    BasicDagColorErrors,
    HashDag
};

/// Rendering mode options
enum class RenderMode
{
    Default,       // Existing rendering
    MarchingCubes, // Marching cubes rendering
};

constexpr uint32 CNumDags = 4;

/// String representation of DAG type
std::string dag_to_string(EDag dag);

/// String representation of tool type
std::string tool_to_string(ETool tool);

/// Main engine class responsible for managing DAGs, camera views, and replay functionality
class Engine
{
public:
    //-------------------------------------------------------------------------
    // Core Engine Members
    //-------------------------------------------------------------------------
    static Engine engine; /// Singleton instance

    /// DAG data structures
    BasicDAG basicDag;
    HashDAG hashDag;
    BasicDAGCompressedColors basicDagCompressedColors;
    BasicDAGUncompressedColors basicDagUncompressedColors;
    BasicDAGColorErrors basicDagColorErrors;
    HashDAGColors hashDagColors;
    HashDAGUndoRedo undoRedo;
    DAGInfo dagInfo;

    /// Rendering configuration
    RenderMode renderMode = RenderMode::Default;

    //-------------------------------------------------------------------------
    // Camera & View Management
    //-------------------------------------------------------------------------
    CameraView view;            /// Current camera view
    CameraView targetView;      /// Target camera view for smooth transitions
    CameraView initialView;     /// Initial camera view for transitions
    bool moveToTarget = false;  /// Flag to enable camera transition
    double targetLerpTime = 0;  /// Current interpolation time
    double targetLerpSpeed = 1; /// Speed of camera transition

    /// Initialize camera transition to target view
    inline void init_target_lerp()
    {
        moveToTarget = true;
        initialView = view;
        targetLerpTime = 0;
    }

    //-------------------------------------------------------------------------
    // Replay & Recording
    //-------------------------------------------------------------------------
    ReplayManager replayReader;  /// Manages replay loading and playback
    VideoManager videoManager;   /// Manages video recording and playback
    StatsRecorder statsRecorder; /// Records statistics for benchmarking

    //-------------------------------------------------------------------------
    // Configuration & State
    //-------------------------------------------------------------------------
    /// Structure to store configuration options for editing and tool use
    struct EditConfig
    {
        EDag currentDag = EDag::HashDag;
        ETool tool = ETool::Sphere;
        EDebugColors debugColors = EDebugColors::None;
        uint32 debugColorsIndexLevel = 0;
        float radius = 10;
        uint3 copySourcePath = make_uint3(0, 0, 0);
        uint3 copyDestPath = make_uint3(0, 0, 0);
        uint3 path;
    };
    EditConfig config;

    //-------------------------------------------------------------------------
    // Public Interface Methods
    //-------------------------------------------------------------------------
    /// Initialize the engine with headless mode flag
    void init(bool headLess);

    /// Run the main loop (either graphics or headless)
    void loop();

    /// Clean up resources
    void destroy();

    /// Set the current DAG type
    void set_dag(EDag dag);

    /// Toggle between fullscreen and windowed mode
    void toggle_fullscreen();

    /// Template function to perform editing on the DAG
    template <typename T, typename... TArgs>
    void edit(TArgs &&...Args)
    {
        PROFILE_FUNCTION();

        lastEditTimestamp = statsRecorder.get_frame_timestamp();
        lastEditFrame = frameIndex;

        BasicStats stats;

        /// is disabled inside the function
        hashDag.data.prefetch();

        stats.start_work("creating edit tool");
        auto tool = T(std::forward<TArgs>(Args)...);
        stats.flush(statsRecorder);

        stats.start_work("total edits");
        hashDag.edit_threads(tool, hashDagColors, undoRedo, statsRecorder);
        stats.flush(statsRecorder);

        stats.start_work("upload_to_gpu");
        hashDag.data.upload_to_gpu();
        stats.flush(statsRecorder);

        statsRecorder.report("radius", tool.radius);
    }

    // Marching Cubes related methods
    void setupMarchingCubes();
    void updateMarchingCubes();
    void renderMarchingCubes();
    void cleanupMarchingCubes();

    // Marching Cubes functions
    std::vector<float> marchingCubes(const std::vector<float> &field, int gridSize, float isoLevel);
    std::vector<float> generateHashDAGField(int gridSize, float scale);

private:
    //-------------------------------------------------------------------------
    // Input & State Management
    //-------------------------------------------------------------------------
    /// Internal structure to track the input state (keyboard, mouse, etc.)
    struct InputState
    {
        std::vector<bool> keys = std::vector<bool>(GLFW_KEY_LAST + 1, false);
        std::vector<bool> mouse = std::vector<bool>(8, false);
        double mousePosX = 0;
        double mousePosY = 0;
    };
    InputState state; /// Current input state

    /// Input callback handlers
    void key_callback_impl(int key, int scancode, int action, int mods);
    void mouse_callback_impl(int button, int action, int mods);
    void scroll_callback_impl(double xoffset, double yoffset);

    /// Static callback functions for GLFW events
    static void key_callback(GLFWwindow *, int key, int scancode, int action, int mods)
    {
        Engine::engine.key_callback_impl(key, scancode, action, mods);
    }
    static void mouse_callback(GLFWwindow *, int button, int action, int mods)
    {
        Engine::engine.mouse_callback_impl(button, action, mods);
    }
    static void scroll_callback(GLFWwindow *, double xoffset, double yoffset)
    {
        Engine::engine.scroll_callback_impl(xoffset, yoffset);
    }

    //-------------------------------------------------------------------------
    // DAG Management
    //-------------------------------------------------------------------------
    bool is_dag_valid(EDag dag) const;
    void next_dag();
    void previous_dag();

    //-------------------------------------------------------------------------
    // Graphics Resources
    //-------------------------------------------------------------------------
    GLFWwindow *window = nullptr;       /// GLFW window handle
    GLuint image = 0;                   /// OpenGL texture handle
    GLuint programID = 0;               /// OpenGL program ID
    GLint textureID = 0;                /// OpenGL texture ID
    GLuint fsvao = 0;                   /// OpenGL full-screen quad VAO
    glf::Context *fontctx = nullptr;    /// Font rendering context
    glf::Buffer *dynamicText = nullptr; /// Dynamic text buffer
    glf::Buffer *staticText = nullptr;  /// Static text buffer

    //-------------------------------------------------------------------------
    // Engine State
    //-------------------------------------------------------------------------
    double dt = 0;                         /// Delta time for each frame
    bool headLess = false;                 /// Flag for headless mode
    bool firstReplay = true;               /// Flag for the first replay
    bool printMemoryStats = false;         /// Flag to print memory stats
    bool shadows = true;                   /// Enable shadows
    float shadowBias = 1;                  /// Shadow bias for rendering
    float fogDensity = 0;                  /// Fog density for rendering
    bool showUI = true;                    /// Flag to show the UI
    float swirlPeriod = 100;               /// Swirl period for special effects
    bool enableSwirl = true;               /// Enable swirl effect
    bool fullscreen = false;               /// Fullscreen mode
    Vector3 transformRotation = {0, 0, 0}; /// Camera rotation
    float transformScale = 1;              /// Camera scale
    double time = 0;                       /// Global time
    uint32 frameIndex = 0;                 /// Frame index for tracking frames
    std::unique_ptr<DAGTracer> tracer;     /// DAG tracing functionality
    ReplayManager replayWriter;            /// Manages replay writing

    struct Timings
    {
        double pathsTime = 0;
        double colorsTime = 0;
        double shadowsTime = 0;
        double totalTime = 0;
    };
    Timings timings; /// Timing measurements for profiling

    uint32 lastEditTimestamp = 0; /// Timestamp of last edit operation
    uint32 lastEditFrame = 0;     /// Frame index of last edit operation

    //-------------------------------------------------------------------------
    // Core Engine Update Loop
    //-------------------------------------------------------------------------
    /// Main update function
    void tick();

    /// Core engine subsystems
    void handleCameraAndReplayControls();
    void handleCameraPresets();
    void handleCameraMovement();
    void handleReplayCompletion();

    void processVoxelData();
    double resolveVoxelPaths();
    void updateMousePath();
    double resolveVoxelColors();
    double resolveVoxelShadows();

    void processEditing();
    void applyEditingTool();
    void updateTimingsAndStats();
    void recordReplayStats();

    //-------------------------------------------------------------------------
    // Graphics & Rendering
    //-------------------------------------------------------------------------
    /// Graphics initialization
    void init_graphics();
    void initializeGLFW();
    void initializeGLEW();
    void setupOpenGLState();
    void loadFonts();
    void createFullScreenQuad();

    /// Main loop and frame processing
    void loop_headless();
    void loop_graphics();
    void processFrame();
    void pollInputs();
    void renderFrame();
    void renderMainScene();
    void renderUI();
    void renderUIElements();

    /// Template rendering functions for UI components
    template <typename FormatterFunc>
    void renderToolSpecificUI(FormatterFunc &F, float hx, float dx, float &y);

    template <typename FormatterFunc>
    void renderTimingStatistics(FormatterFunc &F, float hx, float sx, float dx, float &y);

    template <typename FormatterFunc>
    void renderMemoryStatistics(FormatterFunc &F, float hx, float sx, float dx, float &y);

    template <typename FormatterFunc>
    void renderEditStatistics(FormatterFunc &F, float hx, float sx, float dx, float &y);

    /// Check if the application should exit
    bool shouldExitApplication();

    // Marching Cubes related variables
    GLuint marchingCubesVAO = 0;
    GLuint marchingCubesVBO = 0;
    GLuint marchingCubesVertexCount = 0;
    GLuint marchingCubesShaderProgram = 0;

    int m_gridSize = 256;

    // Shader locations
    GLint mcModelLoc = -1;
    GLint mcViewLoc = -1;
    GLint mcProjLoc = -1;
    GLint mcLightPosLoc = -1;
    GLint mcViewPosLoc = -1;
};
