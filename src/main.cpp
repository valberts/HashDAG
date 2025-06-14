#include "typedefs.h"

#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag_factory.h"
#include "dags/dag_utils.h"
#include "engine.h"

/// Entry point for the DAG-based engine. This program initializes the engine, loads data,
/// and runs the main processing loop. It supports various configurations and DAG types.
int main(int argc, char **argv)
{
    PROFILE_FUNCTION();

    // initialize engine singleton
    auto &engine = Engine::engine;

    /// system config
    printf("Using " SCENE "\n");
    printf("%d levels (resolution=%d^3)\n", MAX_LEVELS, 1 << MAX_LEVELS);
#if ENABLE_CHECKS
    std::fprintf(stderr, "CHECKS: ENABLED\n");
#else
    printf("CHECKS: DISABLED\n");
#endif
    printf("IMAGE RESOLUTION: %ux%u\n", imageWidth, imageHeight);

    /// get proper filename
    std::string depth_suffix;
    if (SCENE_DEPTH >= 10) {
        int shift_val = SCENE_DEPTH - 10;
        // The check SCENE_DEPTH >= 10 already ensures shift_val >= 0.
        // Also ensure it's not too large for a 32-bit unsigned shift (typically < 32).
        // SCENE_DEPTH is unlikely to be >= 42, so SCENE_DEPTH - 10 < 32 is probable.
        if (shift_val >= 0 && shift_val < 32) { // Defensive check
            depth_suffix = std::to_string(1u << shift_val) + "k";
        }
        else {
            // Handle unexpected SCENE_DEPTH value for this branch, e.g., if SCENE_DEPTH was massive
            // This case should ideally not be hit if SCENE_DEPTH is within reasonable octree depth limits.
            checkf(false, "Unexpected SCENE_DEPTH %d for filename generation (>=10 branch)", SCENE_DEPTH);
            depth_suffix = "error_depth_k"; // Fallback
        }
    }
    else { // SCENE_DEPTH < 10
        // SCENE_DEPTH is expected to be < 32 and positive here.
        if (SCENE_DEPTH >= 0 && SCENE_DEPTH < 32) { // Defensive check
            depth_suffix = std::to_string(1u << SCENE_DEPTH);
        }
        else {
            // Handle unexpected SCENE_DEPTH value for this branch
            checkf(false, "Unexpected SCENE_DEPTH %d for filename generation (<10 branch)", SCENE_DEPTH);
            depth_suffix = "error_depth"; // Fallback
        }
    }
    const std::string fileName = std::string(SCENE) + depth_suffix;
    //const std::string fileName = std::string(SCENE) + std::to_string(1 << (SCENE_DEPTH - 10)) + "k";

    /// Load uncompressed color data if enabled.
    if (LOAD_UNCOMPRESSED_COLORS)
    {
        BasicDAGFactory::load_uncompressed_colors_from_file(engine.basicDagUncompressedColors, "data/" + fileName + ".basic_dag.uncompressed_colors.bin");
    }
    /// Load compressed color data if enabled.
    if (LOAD_COMPRESSED_COLORS)
    {
        BasicDAGFactory::load_compressed_colors_from_file(engine.basicDagCompressedColors, "data/" + fileName + ".basic_dag.compressed_colors.variable.bin");
    }
    /// Load the BasicDAG structure from a binary file.
    BasicDAGFactory::load_dag_from_file(engine.dagInfo, engine.basicDag, "data/" + fileName + ".basic_dag.dag.bin");

    /// Fix enclosed leaves and re-save compressed colors if required (disabled by default).
#if 0
    DAGUtils::fix_enclosed_leaves(engine.basicDag, engine.basicDagCompressedColors.enclosedLeaves, engine.basicDagCompressedColors.topLevels);
#if 0
	BasicDAGFactory::save_compressed_colors_to_file(engine.basicDagCompressedColors, "data/" FILENAME ".basic_dag.compressed_colors.variable.bin");
    engine.basicDagCompressedColors.free();
    BasicDAGFactory::load_compressed_colors_from_file(engine.basicDagCompressedColors, "data/" FILENAME ".basic_dag.compressed_colors.variable.bin");
#endif
#endif

    /// If compressed colors are loaded, initialize the HashDAG and its colors from the BasicDAG.
    if (LOAD_COMPRESSED_COLORS)
    {
        HashDAGFactory::load_from_DAG(engine.hashDag, engine.basicDag, 0x8FFFFFFF / C_pageSize / sizeof(uint32));
        HashDAGFactory::load_colors_from_DAG(engine.hashDagColors, engine.basicDag, engine.basicDagCompressedColors);
    }

    engine.basicDagColorErrors.uncompressedColors = engine.basicDagUncompressedColors;
    engine.basicDagColorErrors.compressedColors = engine.basicDagCompressedColors;

    /// engine.basicDag.free();

    // either use basic dag or hash dag
    engine.init(HEADLESS);
#if USE_NORMAL_DAG
    engine.set_dag(EDag::BasicDagCompressedColors);
#else
    engine.set_dag(EDag::HashDag);
#endif

    /// Uncomment this section to toggle fullscreen and load video data (disabled by default).
    // #if USE_VIDEO
    //	engine.toggle_fullscreen();
    //     engine.videoManager.load_video("./videos/" SCENE "_" VIDEO_NAME ".txt");
    //	std::this_thread::sleep_for(std::chrono::seconds(5));
    // #else
    //     engine.replayReader.load_csv("./replays/" SCENE "_" REPLAY_NAME ".csv");
    // #endif

    printf("Starting...\n");

#ifdef PROFILING_PATH
    engine.hashDag.data.save_bucket_sizes(true);
#endif

    engine.loop();
    engine.destroy();

    return 0;
}
