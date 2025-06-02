#include "tracer.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"
#include "dags/dag_utils.h"

// order: (shouldFlipX, shouldFlipY, shouldFlipZ)
DEVICE uint8 next_child(uint8 order, uint8 mask)
{
    for (uint8 child = 0; child < 8; ++child)
    {
        uint8 childInOrder = child ^ order;
        if (mask & (1u << childInOrder))
            return childInOrder;
    }
    check(false);
    return 0;
}

template <typename TDAG>
DEVICE float get_trilinear_density(const TDAG &dag, float3 world_point)
{
    // Shift world_point to align interpolation grid with voxel centers
    float3 sample_point = world_point - make_float3(0.5f, 0.5f, 0.5f);

    // Determine the integer coordinates of the bottom-left-front "anchor" corner
    // of the 1x1x1 conceptual cube that sample_point falls within.
    float3 floor_sample_point = make_float3(
        floorf(sample_point.x), // Use floorf for float, or floor for double
        floorf(sample_point.y),
        floorf(sample_point.z));
    uint3 base_voxel_coord = make_uint3(
        static_cast<unsigned int>(floor_sample_point.x),
        static_cast<unsigned int>(floor_sample_point.y),
        static_cast<unsigned int>(floor_sample_point.z));

    // Get occupancy values for the 8 corners of this cell
    float V[8];
    for (int i = 0; i < 8; ++i)
    {
        Path cornerPath(make_uint3(
            base_voxel_coord.x + ((i & 1) ? 1 : 0),
            base_voxel_coord.y + ((i & 2) ? 1 : 0),
            base_voxel_coord.z + ((i & 4) ? 1 : 0)));
        V[i] = DAGUtils::get_value(dag, cornerPath) ? 1.0f : 0.0f;
    }

    // Calculate local coordinates (weights) within this 1x1x1 cell
    // These are now relative to the corner of the cell containing the shifted sample_point
    float3 local_coords = sample_point - floor_sample_point;

    // Trilinear interpolation (same formula as before)
    float u = local_coords.x;
    float v = local_coords.y;
    float w = local_coords.z;

    float c00 = V[0] * (1 - u) + V[1] * u;
    float c10 = V[2] * (1 - u) + V[3] * u;
    float c01 = V[4] * (1 - u) + V[5] * u;
    float c11 = V[6] * (1 - u) + V[7] * u;

    float c0 = c00 * (1 - v) + c10 * v;
    float c1 = c01 * (1 - v) + c11 * v;

    return c0 * (1 - w) + c1 * w;
}

DEVICE bool ray_box_intersect(
    const float3 &ray_origin,
    const float3 &ray_direction,
    const float3 &ray_direction_inverse,
    const float3 &box_min,
    const float3 &box_max,
    float &t_entry, // out parameter for the t-value of entry into the AABB
    float &t_exit   // out parameter for the t-value of exit from the AABB
)
{
    // 1. Calculate inverse direction: Pre-calculating 1/direction avoids repeated divisions.
    // float3 inv_dir = make_float3(1.0f / ray_direction.x, 1.0f / ray_direction.y, 1.0f / ray_direction.z);
    // Note: This can lead to issues if ray_direction components are zero.
    // Robust implementations handle this (e.g., if ray_direction.x is 0, and ray_origin.x is not between box_min.x and box_max.x, then no intersection with X-slab).
    // For simplicity, many basic implementations assume non-zero direction components or that the logic handles infinities correctly.

    // 2. Calculate t-values for intersection with planes of each slab
    //    (box_plane_coord - ray_origin_coord) * inv_dir_coord
    float3 t_min_planes = (box_min - ray_origin) * ray_direction_inverse; // t-values for hitting the 'min' faces (xmin, ymin, zmin)
    float3 t_max_planes = (box_max - ray_origin) * ray_direction_inverse; // t-values for hitting the 'max' faces (xmax, ymax, zmax)

    // 3. Determine entry and exit t for each slab (dimension)
    //    If inv_dir.x is negative, t_min_planes.x will be for the max_x plane and t_max_planes.x for min_x plane.
    //    So, min() and max() correctly find the actual entry and exit for the slab regardless of ray direction.
    float t_enter_x = min(t_min_planes.x, t_max_planes.x);
    float t_exit_x = max(t_min_planes.x, t_max_planes.x);

    float t_enter_y = min(t_min_planes.y, t_max_planes.y);
    float t_exit_y = max(t_min_planes.y, t_max_planes.y);

    float t_enter_z = min(t_min_planes.z, t_max_planes.z);
    float t_exit_z = max(t_min_planes.z, t_max_planes.z);

    // 4. Find the overall latest entry point and earliest exit point
    t_entry = max(max(t_enter_x, t_enter_y), t_enter_z); // Latest entry into any slab
    t_exit = min(min(t_exit_x, t_exit_y), t_exit_z);     // Earliest exit from any slab

    // 5. Perform the intersection test
    //    - t_entry < t_exit: The entry point must be before the exit point for an overlap.
    //    - t_exit >= 0.0f: The box must not be entirely behind the ray's origin.
    //      (If t_exit < 0, the entire intersection is behind the ray).
    return t_entry < t_exit && t_exit >= 0.0f;
}
template <bool isRoot, typename TDAG>
DEVICE uint8 compute_intersection_mask(
    uint32 level,
    const Path &path,
    const TDAG &dag,
    const float3 &rayOrigin,
    const float3 &rayDirection,
    const float3 &rayDirectionInverted)
{
    // Find node center = .5 * (boundsMin + boundsMax) + .5f
    const uint32 shift = dag.levels - level;

    const float radius = float(1u << (shift - 1));
    const float3 center = make_float3(radius) + path.as_position(shift);

    const float3 centerRelativeToRay = center - rayOrigin;

    // Ray intersection with axis-aligned planes centered on the node
    // => rayOrg + tmid * rayDir = center
    const float3 tmid = centerRelativeToRay * rayDirectionInverted;

    // t-values for where the ray intersects the slabs centered on the node
    // and extending to the side of the node
    float tmin, tmax;
    {
        const float3 slabRadius = radius * abs(rayDirectionInverted);
        const float3 pmin = tmid - slabRadius;
        tmin = max(max(pmin), .0f);

        const float3 pmax = tmid + slabRadius;
        tmax = min(pmax);
    }

    // Check if we actually hit the root node
    // This test may not be entirely safe due to float precision issues.
    // especially on lower levels. For the root node this seems OK, though.
    if (isRoot && (tmin >= tmax))
    {
        return 0;
    }

    // Identify first child that is intersected
    // NOTE: We assume that we WILL hit one child, since we assume that the
    //       parents bounding box is hit.
    // NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
    //       intersection point, since this point might lie too close to an
    //       axis plane. Instead, we use the midpoint between max and min which
    //       will lie in the correct node IF the ray only intersects one node.
    //       Otherwise, it will still lie in an intersected node, so there are
    //       no false positives from this.
    uint8 intersectionMask = 0;
    {
        const float3 pointOnRay = (0.5f * (tmin + tmax)) * rayDirection;

        uint8 const firstChild =
            ((pointOnRay.x >= centerRelativeToRay.x) ? 4 : 0) +
            ((pointOnRay.y >= centerRelativeToRay.y) ? 2 : 0) +
            ((pointOnRay.z >= centerRelativeToRay.z) ? 1 : 0);

        intersectionMask |= (1u << firstChild);
    }

    // We now check the points where the ray intersects the X, Y and Z plane.
    // If the intersection is within (ray_tmin, ray_tmax) then the intersection
    // point implies that two voxels will be touched by the ray. We find out
    // which voxels to mask for an intersection point at +X, +Y by setting
    // ALL voxels at +X and ALL voxels at +Y and ANDing these two masks.
    //
    // NOTE: When the intersection point is close enough to another axis plane,
    //       we must check both sides or we will get robustness issues.
    const float epsilon = 1e-4f;

    if (tmin <= tmid.x && tmid.x <= tmax)
    {
        const float3 pointOnRay = tmid.x * rayDirection;

        uint8 A = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            A |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            A |= 0x33;

        uint8 B = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            B |= 0xAA;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            B |= 0x55;

        intersectionMask |= A & B;
    }
    if (tmin <= tmid.y && tmid.y <= tmax)
    {
        const float3 pointOnRay = tmid.y * rayDirection;

        uint8 C = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            C |= 0xF0;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            C |= 0x0F;

        uint8 D = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            D |= 0xAA;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            D |= 0x55;

        intersectionMask |= C & D;
    }
    if (tmin <= tmid.z && tmid.z <= tmax)
    {
        const float3 pointOnRay = tmid.z * rayDirection;

        uint8 E = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            E |= 0xF0;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            E |= 0x0F;

        uint8 F = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            F |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            F |= 0x33;

        intersectionMask |= E & F;
    }

    return intersectionMask;
}

struct StackEntry
{
    uint32 index;
    uint8 childMask;
    uint8 visitMask;
};

template <typename TDAG>
__global__ void Tracer::trace_paths(const TracePathsParams traceParams, const TDAG dag)
{
    // Target pixel coordinate
    const uint2 pixel = make_uint2(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside.

    // Pre-calculate per-pixel data
    const float3 rayOrigin = make_float3(traceParams.cameraPosition);
    const float3 rayDirection = make_float3(normalize(traceParams.rayMin + pixel.x * traceParams.rayDDx + pixel.y * traceParams.rayDDy - traceParams.cameraPosition));

    const float3 rayDirectionInverse = make_float3(make_double3(1. / rayDirection.x, 1. / rayDirection.y, 1. / rayDirection.z));
    const uint8 rayChildOrder =
        (rayDirection.x < 0.f ? 4 : 0) +
        (rayDirection.y < 0.f ? 2 : 0) +
        (rayDirection.z < 0.f ? 1 : 0);
    // --- Outer loop for attempting multiple hits ---
    // This loop allows us to continue searching if a ray march fails
    // but the DAG traversal had found a coarse hit.
    const int MAX_SEE_THROUGH_ATTEMPTS = pow(2, dag.levels); // potentially pierce through every voxel in a straight line

    Path final_stored_path(0, 0, 0);
    Path final_stored_preHitPath(0, 0, 0);
    float final_hit_t = -1.0f; // Will store the t-value of the *successful* smooth hit

    // Traversal state needs to be managed carefully across attempts.
    // For a simple "continue from where we left off", we need to modify
    // the main DAG traversal's visitMasks.
    // A simpler (but potentially less efficient for many see-throughs)
    // way is to restart traversal but tell it to ignore previous hit voxels.
    // For now, let's illustrate a more integrated approach.

    // Initial DAG traversal setup (done once before the see-through loop)
    uint32 initial_level_state = 0;
    Path initial_path_state(0, 0, 0);
    StackEntry initial_stack_state[MAX_LEVELS]; // Store the full stack
    StackEntry initial_cache_state;
    initial_cache_state.index = dag.get_first_node_index();
    initial_cache_state.childMask = Utils::child_mask(dag.get_node(0, initial_cache_state.index));
    initial_cache_state.visitMask = initial_cache_state.childMask & compute_intersection_mask<true>(0, initial_path_state, dag, rayOrigin, rayDirection, rayDirectionInverse);

    for (int attempt = 0; attempt < MAX_SEE_THROUGH_ATTEMPTS; ++attempt)
    {
        // State for the current traversal attempt
        uint32 level = initial_level_state; // Start from root or last good point
        Path path = initial_path_state;     // Current path in DAG
        Path preHitPath(0, 0, 0);           // Parent of the voxel to be ray marched

        StackEntry stack[MAX_LEVELS];
        for (int i = 0; i < MAX_LEVELS; ++i)
            stack[i] = initial_stack_state[i]; // Copy stack

        StackEntry cache = initial_cache_state; // Current node's cache
        Leaf cachedLeaf;

        // If it's not the first attempt, we will continue the traversal from where the last attempt left off

        // --- DAG TRAVERSAL ---
        for (;;) // Inner DAG traversal loop
        {
            // Ascend if there are no children left.

            {
                uint32 newLevel = level;
                while (newLevel > 0 && !cache.visitMask) // If no more visitable children at current level
                {
                    newLevel--;
                    cache = stack[newLevel]; // Pop parent's state from stack
                }
                if (newLevel == 0 && !cache.visitMask) // Back at root and still no visitable children
                {
                    // Ray truly misses the rest of the DAG
                    if (attempt == 0)
                    {                                            // Only if first attempt also misses everything
                        final_stored_path = Path(0, 0, 0);       // Mark path as null
                        final_stored_preHitPath = Path(0, 0, 0); // Mark preHitPath as null
                    }
                    // JUMP to the end of all attempts, as no more surfaces can be found
                    goto end_attempts; // EXIT 1: Ray misses the entire DAG
                }
                path.ascend(level - newLevel);
                level = newLevel;
            }
            // Find next child in order by the current ray's direction
            const uint8 nextChild = next_child(rayChildOrder, cache.visitMask);
            cache.visitMask &= ~(1u << nextChild); // Mark as visited for THIS parent

            // Store parent state *before* modifying 'cache' for descent
            StackEntry parent_cache_for_stack = cache;

            // BEFORE descending into what MIGHT be the final voxel:
            if (level + 1 == dag.levels)
            {
                // 'path' currently refers to the PARENT of the voxel we are about to hit.
                preHitPath = path;
            }
            path.descend(nextChild);
            stack[level] = parent_cache_for_stack; // Push parent's state (with its updated visitMask)
            level++;

            // If we're at the final level, we have intersected a single voxel.
            if (level == dag.levels)
            {
                // BREAK from the inner DAG traversal loop.
                // Control flow will proceed to the ray marching section for 'path' and 'preHitPath'.
                break; // EXIT 2: Reached a leaf voxel - DAG traversal complete
            }

            // Are we in an internal node?
            if (level < dag.leaf_level())
            {
                // Update cache using parent info from stack entry (stack[level-1])
                cache.index = dag.get_child_index(level - 1, stack[level - 1].index, stack[level - 1].childMask, nextChild);
                cache.childMask = Utils::child_mask(dag.get_node(level, cache.index));
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
            }
            else // in packed leaf levels
            {
                /* The second-to-last and last levels are different: the data
                 * of these two levels (2^3 voxels) are packed densely into a
                 * single 64-bit word.
                 */
                uint8 childMask_leaf;
                if (level == dag.leaf_level())
                {
                    const uint32 addr = dag.get_child_index(level - 1, stack[level - 1].index, stack[level - 1].childMask, nextChild);
                    cachedLeaf = dag.get_leaf(addr);
                    childMask_leaf = cachedLeaf.get_first_child_mask();
                }
                else
                {
                    childMask_leaf = cachedLeaf.get_second_child_mask(nextChild);
                }
                // No need to set the index for bottom nodes
                cache.childMask = childMask_leaf;
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
            }
        } // End of inner DAG traversal loop

        // if (!dag_hit_this_attempt)
        // {
        //     // Should have been caught by the ascend logic if no DAG hit possible
        //     if (attempt == 0)
        //     {
        //         final_stored_path = Path(0, 0, 0);
        //         final_stored_preHitPath = Path(0, 0, 0);
        //     }
        //     goto end_attempts;
        // }

        // --------------------- RAY MARCHING ----------------------------
        // The DAG traversal above found a coarse voxel hit (stored in 'path')
        // and its parent (stored in 'preHitPath'). Now we refine this hit
        // by marching along the ray within a small local region.

        // STEP 1: Define Local Marching Bounding Box
        // ------------------------------------------
        // 1a. Get the world-space corner of the 2x2x2 parent cell identified by 'preHitPath'.
        //     The '1' in as_position(1) likely relates to the level difference for a 2x2x2 cell.
        float3 bounds_min = preHitPath.as_position(1);
        // 1b. Define the max corner of this 2x2x2 cell.
        float3 bounds_max = make_float3(
            bounds_min.x + 2.0f,
            bounds_min.y + 2.0f,
            bounds_min.z + 2.0f);

        // 1c. Expand the box by one voxel in all directions.
        //     This helps find surfaces that might be slightly outside the initial 2x2x2 cell,
        //     especially due to trilinear interpolation effects or if the hit voxel is at an edge.
        bounds_min = make_float3(bounds_min.x - 1.0f, bounds_min.y - 1.0f, bounds_min.z - 1.0f);
        bounds_max = make_float3(bounds_max.x + 1.0f, bounds_max.y + 1.0f, bounds_max.z + 1.0f);

        // STEP 2: Intersect Ray with Local Marching Box
        // ---------------------------------------------
        float march_t_start, march_t_end;
        bool can_march = ray_box_intersect(rayOrigin, rayDirection, rayDirectionInverse,
                                           bounds_min, bounds_max,
                                           march_t_start, march_t_end);

        // Initialize hit_t to -1.0 (no hit found yet).
        float current_hit_t = -1.0f; // t for this specific ray march attempt

        // STEP 3: Initialize Ray Marching Parameters
        // ------------------------------------------
        if (can_march)
        {
            // Start marching from the entry point into the local box.
            float current_t = march_t_start;
            float step_size = 0.25f;
            int max_steps = static_cast<int>((march_t_end - march_t_start) / step_size) + 5;
            max_steps = min(max_steps, 100);
            float density_threshold = 0.5f;
            float3 current_pos_march = rayOrigin + current_t * rayDirection;
            float prev_density = get_trilinear_density(dag, current_pos_march);

            // STEP 4: March Along the Ray within the Local Box
            // -----------------------------------------------
            while (step_size > 0.01f)
            {
                current_t += step_size;
                if (current_t > march_t_end)
                {
                    break;
                }
                current_pos_march = rayOrigin + current_t * rayDirection;
                float current_density = get_trilinear_density(dag, current_pos_march);
                if (prev_density < density_threshold && current_density >= density_threshold)
                {
                    current_hit_t = current_t;
                    current_t -= step_size;
                    step_size *= 0.5f;
                    continue;
                }
                prev_density = current_density;
            }
            // for (int i = 0; i < max_steps; ++i)
            // {
            //     current_t += step_size;
            //     if (current_t > march_t_end) // outside of bounding box, stop
            //         break;
            //     // Get initial density at the starting point of the march.
            //     current_pos_march = rayOrigin + current_t * rayDirection;
            //     float current_density = get_trilinear_density(dag, current_pos_march);
            //     if ((prev_density < density_threshold && current_density >= density_threshold) ||
            //         (prev_density >= density_threshold && current_density < density_threshold))
            //     {
            //         if (abs(current_density - prev_density) > 1e-6f)
            //         {
            //             float t_interp_fraction = (density_threshold - prev_density) / (current_density - prev_density);
            //             current_hit_t = (current_t - step_size) + t_interp_fraction * step_size;
            //         }
            //         else
            //         {
            //             current_hit_t = current_t;
            //         }
            //         // we hit something, stop
            //         break;
            //     }
            //     prev_density = current_density;
            // }
        } // end if(can_march)

        // STEP 5: Handle Ray Marching Result
        // ----------------------------------
        // --- DECIDE whether to store this hit or continue ---
        if (current_hit_t >= 0.0f)
        {
            // Smooth surface found! This is our final hit.
            final_stored_path = path;             // The coarse voxel of this successful march
            final_stored_preHitPath = preHitPath; // Its parent
            final_hit_t = current_hit_t;          // Store the t-value of the smooth hit

            // JUMP to the end of all attempts, as we've found our surface.
            goto end_attempts; // Break out of the outer 'attempt' loop
        }
        else
        {
            // Ray march MISSED for path/preHitPath.
            // We need to prepare the DAG traversal state to continue *past* this failed hit.
            // The 'stack' holds the state of the parents. 'level-1' is the parent of 'path'.
            // The 'visitMask' in 'stack[level-1]' (which was just used to find 'path')
            // has already had the bit for 'path' cleared by the `cache.visitMask &= ~(1u << nextChild);`
            // line *before* `stack[level] = parent_cache_for_stack;`.
            // So, when the next 'attempt' starts, and if it correctly restores `level`, `path`, `stack`, and `cache`
            // to the state of the parent of the *just failed* hit, the `next_child` call
            // should pick the next available sibling.

            // Restore state to the parent of the just-failed hit, so the next iteration
            // of the outer loop can try the next sibling or ascend.
            if (level > 0)
            { // Should always be true if dag_hit_this_attempt was true
                initial_level_state = level - 1;
                initial_path_state = preHitPath; // Path of the parent
                for (int i = 0; i < MAX_LEVELS; ++i)
                    initial_stack_state[i] = stack[i];  // Save current stack
                initial_cache_state = stack[level - 1]; // This cache has the updated visitMask for the parent
            }
            else
            {
                // Should not happen if dag_hit_this_attempt was true.
                // If it does, implies an issue or that the DAG root itself failed the march.
                // Treat as no more hits possible.
                goto end_attempts; // JUMP to the end of all attempts.
            }
            // If initial_cache_state.visitMask is 0, the next attempt's ascend logic will handle it.
        }
    } // End of outer 'attempt' loop

end_attempts:;

    // Store the final result (could be a smooth hit, or null if all attempts failed)
    // If 'goto end_attempts' was taken due to a successful ray march, these will be the hit details.
    // If 'goto end_attempts' was taken due to a total DAG miss, these will be null (or reflect the initial miss).
    // If the loop completes all attempts without a successful ray_march,
    // 'final_stored_path' will be from the last failed DAG hit (if any, otherwise null),
    // and 'final_hit_t' will be -1.0f

    // Ideally, if all attempts fail, final_stored_path should also be null.
    if (final_hit_t < 0.0f)
    { // If no smooth hit was ever found after all attempts
        final_stored_path = Path(0, 0, 0);
        final_stored_preHitPath = Path(0, 0, 0);
    }

    final_stored_path.store(pixel.x, imageHeight - 1 - pixel.y, traceParams.pathsSurface);
    final_stored_preHitPath.store(pixel.x, imageHeight - 1 - pixel.y, traceParams.preHitPathsSurface);
}

template <typename TDAG, typename TDAGColors>
__global__ void Tracer::trace_colors(const TraceColorsParams traceParams, const TDAG dag, const TDAGColors colors)
{
    const uint2 pixel = make_uint2(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    // RECONSTRUCT LIKE IN TRACE_PATHS

    // Pre-calculate per-pixel data
    // const float3 rayOrigin = make_float3(traceParams.cameraPosition);
    // const float3 rayDirection = make_float3(normalize(traceParams.rayMin + pixel.x * traceParams.rayDDx + pixel.y * traceParams.rayDDy - traceParams.cameraPosition));

    // const float3 rayDirectionInverse = make_float3(make_double3(1. / rayDirection.x, 1. / rayDirection.y, 1. / rayDirection.z));
    // const uint8 rayChildOrder =
    //     (rayDirection.x < 0.f ? 4 : 0) +
    //     (rayDirection.y < 0.f ? 2 : 0) +
    //     (rayDirection.z < 0.f ? 1 : 0);

    const auto setColorImpl = [&](uint32 color)
    {
        surf2Dwrite(color, traceParams.colorsSurface, (int)sizeof(uint32) * pixel.x, pixel.y, cudaBoundaryModeClamp);
    };
    // const auto setColorImpl = [&](uint32 color)
    // {
    //     surf2Dwrite(color, traceParams.colorsSurface, (int)sizeof(uint32) * pixel.x, imageHeight - 1 - pixel.y, cudaBoundaryModeClamp);
    // };

    // ---------------------- RENDER WHITE GEOMETRY ---------------------------
    // const Path path = Path::load(pixel.x, pixel.y, traceParams.pathsSurface); // Original hit path
    // if (path.is_null())
    // {
    //     setColorImpl(ColorUtils::float3_to_rgb888(make_float3(187, 242, 250) / 255.f));
    //     return;
    // }
    // else
    // {
    //     setColorImpl(ColorUtils::float3_to_rgb888(make_float3(255, 255, 255) / 255.f));
    //     return;
    // }

    // ---------------- DISABLED -----------------------
    // ---------------- RAY MARCHING TO FIND SMOOTH SURFACE ----------------------

    // // const Path preHitPath = Path::load(pixel.x, pixel.y, traceParams.preHitPathsSurface);
    // const Path preHitPath = Path::load(pixel.x, imageHeight - 1 - pixel.y, traceParams.preHitPathsSurface);

    // if (preHitPath.is_null())
    // {
    //     setColorImpl(ColorUtils::float3_to_rgb888(make_float3(187, 242, 250) / 255.f)); // Background
    //     return;
    // }

    // // bounds of prehitpath box
    // float3 bounds_min = preHitPath.as_position(1);
    // float3 bounds_max = make_float3(
    //     bounds_min.x + 2.0f,
    //     bounds_min.y + 2.0f,
    //     bounds_min.z + 2.0f);

    // // increase it by 1 voxel so we can find a surface outside a fully filled 8-voxel neighborhood
    // bounds_min = make_float3(bounds_min.x - 1.0f, bounds_min.y - 1.0f, bounds_min.z - 1.0f);
    // bounds_max = make_float3(bounds_max.x + 1.0f, bounds_max.y + 1.0f, bounds_max.z + 1.0f);

    // float march_t_start, march_t_end;

    // if (!ray_box_intersect(rayOrigin, rayDirection, rayDirectionInverse,
    //                        bounds_min, bounds_max,
    //                        march_t_start, march_t_end))
    // {
    //     // Ray completely misses even the expanded marching volume around preHitPath.
    //     // This should be rare if preHitPath was valid, but possible if preHitPath is near edge of scene.
    //     setColorImpl(ColorUtils::float3_to_rgb888(make_float3(1.0f, 0.0f, 1.0f))); // Magenta for miss
    //     return;
    // }

    // // ... (after calculating march_t_start and march_t_end, and after ray_box_intersect succeeded) ...

    // float current_t = march_t_start;
    // // Step size should be less than 1.0 (finest voxel size).
    // // A smaller step size gives more accuracy but is slower. Start with something reasonable.
    // float step_size = 0.25f;                                                         // e.g., 1/4 of a voxel
    // int max_steps = static_cast<int>((march_t_end - march_t_start) / step_size) + 5; // Dynamic max steps + buffer
    // max_steps = min(max_steps, 100);                                                 // Absolute cap for safety

    // float hit_t = -1.0f;
    // float density_threshold = 0.5f;

    // float3 current_pos = rayOrigin + current_t * rayDirection;
    // float prev_density = get_trilinear_density(dag, current_pos);

    // for (int i = 0; i < max_steps; ++i)
    // {
    //     current_t += step_size;
    //     // Check if we've marched beyond our defined end point for this local region
    //     if (current_t > march_t_end)
    //     {
    //         break;
    //     }

    //     current_pos = rayOrigin + current_t * rayDirection;
    //     float current_density = get_trilinear_density(dag, current_pos);

    //     // Check for crossing the density threshold
    //     if ((prev_density < density_threshold && current_density >= density_threshold) ||
    //         (prev_density >= density_threshold && current_density < density_threshold))
    //     {
    //         // We crossed! Refine hit_t using linear interpolation between previous and current sample
    //         if (abs(current_density - prev_density) > 1e-6f)
    //         { // Avoid division by zero
    //             float t_interp_fraction = (density_threshold - prev_density) / (current_density - prev_density);
    //             hit_t = (current_t - step_size) + t_interp_fraction * step_size;
    //         }
    //         else
    //         {
    //             hit_t = current_t; // Densities are too close, just take the current one
    //         }
    //         break; // Found hit
    //     }
    //     prev_density = current_density;
    // }

    // if (hit_t >= 0.0f)
    // {
    //     float3 smoothHitPosition = rayOrigin + hit_t * rayDirection;
    //     // Visualize the hit position (scaled for visibility)
    //     setColorImpl(ColorUtils::float3_to_rgb888(
    //         make_float3(smoothHitPosition.x, smoothHitPosition.y, smoothHitPosition.z) * 0.0001f // Adjust scale as needed
    //         ));
    // }
    // else
    // {
    //     // No smooth surface hit within the marching distance
    //     // For debugging, you could color this differently than the absolute background
    //     // setColorImpl(ColorUtils::float3_to_rgb888(make_float3(187, 242, 250) / 255.f)); // Background

    //     setColorImpl(ColorUtils::float3_to_rgb888(make_float3(0.1f, 0.2f, 0.1f))); // Dark green for march miss
    // }
    // return; // Stop here for this visualization

    // ------------------------- END RAY MARCHING TO FIND SMOOTH SURFACE ----------------------------

    // Original trace_colors logic (now effectively disabled by the return above)
    const Path path = Path::load(pixel.x, pixel.y, traceParams.pathsSurface); // Original hit path
    if (path.is_null())
    {
        setColorImpl(ColorUtils::float3_to_rgb888(make_float3(187, 242, 250) / 255.f));
        return;
    }

    const float toolStrength = traceParams.toolInfo.strength(path);
    const auto setColor = [&](uint32 color)
    {
#if TOOL_OVERLAY
        if (toolStrength > 0)
        {
            color = ColorUtils::float3_to_rgb888(lerp(ColorUtils::rgb888_to_float3(color), make_float3(1, 0, 0), clamp(100 * toolStrength, 0.f, .5f)));
        }
#endif
        setColorImpl(color);
    };

    const auto invalidColor = [&]()
    {
        uint32 b = (path.path.x ^ path.path.y ^ path.path.z) & 0x1;
        setColor(ColorUtils::float3_to_rgb888(make_float3(1, b, 1.f - b)));
    };

    uint64 nof_leaves = 0;
    uint32 debugColorsIndex = 0;

    uint32 colorNodeIndex = 0;
    typename TDAGColors::ColorLeaf colorLeaf = colors.get_default_leaf();

    uint32 level = 0;
    uint32 nodeIndex = dag.get_first_node_index();
    while (level < dag.leaf_level())
    {
        level++;

        // Find the current childmask and which subnode we are in
        const uint32 node = dag.get_node(level - 1, nodeIndex);
        const uint8 childMask = Utils::child_mask(node);
        const uint8 child = path.child_index(level, dag.levels);

        // Make sure the node actually exists
        if (!(childMask & (1 << child)))
        {
            setColor(0xFF00FF);
            return;
        }

        ASSUME(level > 0);
        if (level - 1 < colors.get_color_tree_levels())
        {
            colorNodeIndex = colors.get_child_index(level - 1, colorNodeIndex, child);
            if (level == colors.get_color_tree_levels())
            {
                check(nof_leaves == 0);
                colorLeaf = colors.get_leaf(colorNodeIndex);
            }
            else
            {
                // TODO nicer interface
                if (!colorNodeIndex)
                {
                    invalidColor();
                    return;
                }
            }
        }

        // Debug
        if (traceParams.debugColors == EDebugColors::Index ||
            traceParams.debugColors == EDebugColors::Position ||
            traceParams.debugColors == EDebugColors::ColorTree)
        {
            if (traceParams.debugColors == EDebugColors::Index &&
                traceParams.debugColorsIndexLevel == level - 1)
            {
                debugColorsIndex = nodeIndex;
            }
            if (level == dag.leaf_level())
            {
                if (traceParams.debugColorsIndexLevel == dag.leaf_level())
                {
                    check(debugColorsIndex == 0);
                    const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
                    debugColorsIndex = childIndex;
                }

                if (traceParams.debugColors == EDebugColors::Index)
                {
                    setColor(Utils::murmurhash32(debugColorsIndex));
                }
                else if (traceParams.debugColors == EDebugColors::Position)
                {
                    constexpr uint32 checkerSize = 0x7FF;
                    float color = ((path.path.x ^ path.path.y ^ path.path.z) & checkerSize) / float(checkerSize);
                    color = (color + 0.5) / 2;
                    setColor(ColorUtils::float3_to_rgb888(Utils::has_flag(nodeIndex) ? make_float3(color, 0, 0) : make_float3(color)));
                }
                else
                {
                    check(traceParams.debugColors == EDebugColors::ColorTree);
                    const uint32 offset = dag.levels - colors.get_color_tree_levels();
                    const float color = ((path.path.x >> offset) ^ (path.path.y >> offset) ^ (path.path.z >> offset)) & 0x1;
                    setColor(ColorUtils::float3_to_rgb888(make_float3(color)));
                }
                return;
            }
            else
            {
                nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
                continue;
            }
        }

        //////////////////////////////////////////////////////////////////////////
        // Find out how many leafs are in the children preceding this
        //////////////////////////////////////////////////////////////////////////
        // If at final level, just count nof children preceding and exit
        if (level == dag.leaf_level())
        {
            for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild)
            {
                if (childMask & (1u << childBeforeChild))
                {
                    const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
                    const Leaf leaf = dag.get_leaf(childIndex);
                    nof_leaves += Utils::popcll(leaf.to_64());
                }
            }
            const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
            const Leaf leaf = dag.get_leaf(childIndex);
            const uint8 leafBitIndex =
                (((path.path.x & 0x1) == 0) ? 0 : 4) |
                (((path.path.y & 0x1) == 0) ? 0 : 2) |
                (((path.path.z & 0x1) == 0) ? 0 : 1) |
                (((path.path.x & 0x2) == 0) ? 0 : 32) |
                (((path.path.y & 0x2) == 0) ? 0 : 16) |
                (((path.path.z & 0x2) == 0) ? 0 : 8);
            nof_leaves += Utils::popcll(leaf.to_64() & ((uint64(1) << leafBitIndex) - 1));

            break;
        }
        else
        {
            ASSUME(level > 0);
            if (level > colors.get_color_tree_levels())
            {
                // Otherwise, fetch the next node (and accumulate leaves we pass by)
                for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild)
                {
                    if (childMask & (1u << childBeforeChild))
                    {
                        const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
                        const uint32 childNode = dag.get_node(level, childIndex);
                        nof_leaves += colors.get_leaves_count(level, childNode);
                    }
                }
            }
            nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
        }
    }

    if (!colorLeaf.is_valid() || !colorLeaf.is_valid_index(nof_leaves))
    {
        invalidColor();
        return;
    }

    auto compressedColor = colorLeaf.get_color(nof_leaves);
    uint32 color =
        traceParams.debugColors == EDebugColors::ColorBits
            ? compressedColor.get_debug_hash()
            : ColorUtils::float3_to_rgb888(
                  traceParams.debugColors == EDebugColors::MinColor
                      ? compressedColor.get_min_color()
                  : traceParams.debugColors == EDebugColors::MaxColor
                      ? compressedColor.get_max_color()
                  : traceParams.debugColors == EDebugColors::Weight
                      ? make_float3(compressedColor.get_weight())
                      : compressedColor.get_color());
    setColor(color);
}

template <typename TDAG>
inline __device__ bool intersect_ray_node_out_of_order(const TDAG &dag, const float3 rayOrigin, const float3 rayDirection)
{
    const float3 rayDirectionInverse = make_float3(make_double3(1. / rayDirection.x, 1. / rayDirection.y, 1. / rayDirection.z));

    // State
    uint32 level = 0;
    Path path(0, 0, 0);

    StackEntry stack[MAX_LEVELS];
    StackEntry cache;
    Leaf cachedLeaf; // needed to iterate on the last few levels

    cache.index = dag.get_first_node_index();
    cache.childMask = Utils::child_mask(dag.get_node(0, cache.index));
    cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, rayOrigin, rayDirection, rayDirectionInverse);

    // Traverse DAG
    for (;;)
    {
        // Ascend if there are no children left.
        {
            uint32 newLevel = level;
            while (newLevel > 0 && !cache.visitMask)
            {
                newLevel--;
                cache = stack[newLevel];
            }

            if (newLevel == 0 && !cache.visitMask)
            {
                path = Path(0, 0, 0);
                break;
            }

            path.ascend(level - newLevel);
            level = newLevel;
        }

        // Find next child in order by the current ray's direction
        const uint8 nextChild = 31 - __clz(cache.visitMask);

        // Mark it as handled
        cache.visitMask &= ~(1u << nextChild);

        // Intersect that child with the ray
        {
            path.descend(nextChild);
            stack[level] = cache;
            level++;

            // If we're at the final level, we have intersected a single voxel.
            if (level == dag.levels)
            {
                return true;
            }

            // Are we in an internal node?
            if (level < dag.leaf_level())
            {
                cache.index = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                cache.childMask = Utils::child_mask(dag.get_node(level, cache.index));
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
            }
            else
            {
                /* The second-to-last and last levels are different: the data
                 * of these two levels (2^3 voxels) are packed densely into a
                 * single 64-bit word.
                 */
                uint8 childMask;

                if (level == dag.leaf_level())
                {
                    const uint32 addr = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                    cachedLeaf = dag.get_leaf(addr);
                    childMask = cachedLeaf.get_first_child_mask();
                }
                else
                {
                    childMask = cachedLeaf.get_second_child_mask(nextChild);
                }

                // No need to set the index for bottom nodes
                cache.childMask = childMask;
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, rayOrigin, rayDirection, rayDirectionInverse);
            }
        }
    }
    return false;
}

// Directed towards the sun
HOST_DEVICE float3 sun_direction()
{
    return normalize(make_float3(0.3f, 1.f, 0.5f));
}

HOST_DEVICE float3 applyFog(float3 rgb,      // original color of the pixel
                            double distance, // camera to point distance
                            double3 rayDir,  // camera to point vector
                            double3 rayOri,
                            float fogDensity) // camera position
{
#if 0
    constexpr float fogDensity = 0.0001f;
    constexpr float c = 1.f;
    constexpr float heightOffset = 20000.f;
    constexpr float heightScale = 1.f;
    double fogAmount = c * exp((heightOffset - rayOri.y * heightScale) * fogDensity) * (1.0 - exp(-distance * rayDir.y * fogDensity)) / rayDir.y;
#else
    fogDensity *= 0.00001f;
    double fogAmount = 1.0 - exp(-distance * fogDensity);
#endif
    double sunAmount = 1.01f * max(dot(rayDir, make_double3(sun_direction())), 0.0);
    float3 fogColor = lerp(make_float3(187, 242, 250) / 255.f, // blue
                           make_float3(1.0f),                  // white
                           float(pow(sunAmount, 30.0)));
    return lerp(rgb, fogColor, clamp(float(fogAmount), 0.f, 1.f));
}

HOST_DEVICE double3 ray_box_intersection(double3 orig, double3 dir, double3 box_min, double3 box_max)
{
    double3 tmin = (box_min - orig) / dir;
    double3 tmax = (box_max - orig) / dir;

    double3 real_min = min(tmin, tmax);
    double3 real_max = max(tmin, tmax);

    // double minmax = min(min(real_max.x, real_max.y), real_max.z);
    double maxmin = max(max(real_min.x, real_min.y), real_min.z);

    // checkf(minmax >= maxmin, "%f > %f", minmax, maxmin);
    return orig + dir * maxmin;
}

template <typename TDAG>
__global__ void Tracer::trace_shadows(const TraceShadowsParams params, const TDAG dag)
{
    const uint2 pixel = make_uint2(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    const auto setColorImpl = [&](float3 color)
    {
        const uint32 finalColor = ColorUtils::float3_to_rgb888(color);
        surf2Dwrite(finalColor, params.colorsSurface, (int)sizeof(uint32) * pixel.x, pixel.y, cudaBoundaryModeClamp);
    };
    const auto setColor = [&](float light, double distance, double3 direction)
    {
        const uint32 colorInt = surf2Dread<uint32>(params.colorsSurface, pixel.x * sizeof(uint32), pixel.y);
        float3 color = ColorUtils::rgb888_to_float3(colorInt);

        color = color * clamp(0.5f + light, 0.f, 1.f);

        color = applyFog(
            color,
            distance,
            direction,
            params.cameraPosition,
            params.fogDensity);

        setColorImpl(color);
    };

    const float3 rayOrigin = make_float3(Path::load(pixel.x, pixel.y, params.pathsSurface).path);
    const double3 cameraRayDirection = normalize(params.rayMin + pixel.x * params.rayDDx + (imageHeight - 1 - pixel.y) * params.rayDDy - params.cameraPosition);

#if EXACT_SHADOWS || PER_VOXEL_FACE_SHADING
    const double3 rayOriginDouble = make_double3(rayOrigin);
    const double3 hitPosition = ray_box_intersection(
        params.cameraPosition,
        cameraRayDirection,
        rayOriginDouble,
        rayOriginDouble + 1);
#endif

#if EXACT_SHADOWS
    const float3 shadowStart = make_float3(hitPosition);
#else
    const float3 shadowStart = rayOrigin;
#endif

#if 0
    setColorImpl(make_float3(clamp_vector(normal, 0, 1)));
    return;
#endif

    if (length(rayOrigin) == 0.0f)
    {
        setColor(1, 1e9, cameraRayDirection);
        return; // Discard cleared or light-backfacing fragments
    }

    const float3 direction = sun_direction();
    const bool isShadowed = intersect_ray_node_out_of_order(dag, shadowStart + params.shadowBias * direction, direction);

    const double3 v = make_double3(rayOrigin) - params.cameraPosition;
    const double distance = length(v);
    const double3 nv = v / distance;

    if (isShadowed)
    {
        setColor(0, distance, nv);
    }
    else
    {
#if PER_VOXEL_FACE_SHADING
        const double3 voxelOriginToHitPosition = normalize(hitPosition - (rayOriginDouble + 0.5));
        const auto truncate_signed = [](double3 d)
        { return make_double3(int32(d.x), int32(d.y), int32(d.z)); };
        const double3 normal = truncate_signed(voxelOriginToHitPosition / max(abs(voxelOriginToHitPosition)));
        setColor(max(0.f, dot(make_float3(normal), sun_direction())), distance, nv);
#else
        setColor(1, distance, nv);
#endif
    }

#if 0 // AO code copy-pasted from Erik's impl, doesn't compile at all
    constexpr int sqrtNofSamples = 8;

    float avgSum = 0;
    for (int y = 0; y < sqrtNofSamples; y++)
    {
        for (int x = 0; x < sqrtNofSamples; x++)
        {
            int2 coord = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
            float3 normal = make_float3(tex2D(normalTexture, float(coord.x), float(coord.y)));
            float3 tangent = normalize3(perp3(normal));
            float3 bitangent = cross(normal, tangent);
            //int2 randomCoord = make_int2((coord.x * sqrtNofSamples + x + randomSeed.x)%RAND_SIZE, (coord.y * sqrtNofSamples + y + randomSeed.y)%RAND_SIZE);
            int2 randomCoord = make_int2((coord.x * sqrtNofSamples + x + randomSeed.x) & RAND_BITMASK, (coord.y * sqrtNofSamples + y + randomSeed.y) & RAND_BITMASK);
            float2 randomSample = tex2D(randomTexture, randomCoord.x, randomCoord.y);
            float randomLength = tex2D(randomTexture, randomCoord.y, randomCoord.x).x;
            float2 dxdy = make_float2(1.0f / float(sqrtNofSamples), 1.0f / float(sqrtNofSamples));
            float3 sample = cosineSampleHemisphere(make_float2(x * dxdy.x, y * dxdy.y) + (1.0 / float(sqrtNofSamples)) * randomSample);
            float3 ray_d = normalize3(sample.x * tangent + sample.y * bitangent + sample.z * normal);
            avgSum += intersectRayNode_outOfOrder<maxLevels>(ray_o, ray_d, ray_tmax * randomLength, rootCenter, rootRadius, coneOpening) ? 0.0f : 1.0f;
        }
    }
    avgSum /= float(sqrtNofSamples * sqrtNofSamples);
#endif
}

template __global__ void Tracer::trace_paths<BasicDAG>(TracePathsParams, BasicDAG);
template __global__ void Tracer::trace_paths<HashDAG>(TracePathsParams, HashDAG);

template __global__ void Tracer::trace_shadows<BasicDAG>(TraceShadowsParams, BasicDAG);
template __global__ void Tracer::trace_shadows<HashDAG>(TraceShadowsParams, HashDAG);

#define COLORS_IMPL(Dag, Colors) \
    template __global__ void Tracer::trace_colors<Dag, Colors>(TraceColorsParams, Dag, Colors);

COLORS_IMPL(BasicDAG, BasicDAGUncompressedColors)
COLORS_IMPL(BasicDAG, BasicDAGCompressedColors)
COLORS_IMPL(BasicDAG, BasicDAGColorErrors)
COLORS_IMPL(HashDAG, HashDAGColors)