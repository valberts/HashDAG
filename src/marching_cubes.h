#pragma once

#include "typedefs.h"
#include "cuda_math.h"

namespace MarchingCubes
{
    extern const int edgeTable[256];
    extern const int triTable[256][16];

    struct MCVertex
    {
        float3 position;
        float3 normal;
    };

    struct MCMesh
    {
        std::vector<MCVertex> vertices;
        std::vector<uint32> indices;

        void clear()
        {
            vertices.clear();
            indices.clear();
        }
    };

    template <typename TDAG>
    void generate_mesh(const TDAG& dag, MCMesh& mesh, float isoLevel = 0.5f);
}