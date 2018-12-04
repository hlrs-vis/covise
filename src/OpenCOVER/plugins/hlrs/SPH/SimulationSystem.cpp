/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstdlib>
#include <fstream>
#include <algorithm>

//#include "Fluid.h"

//#include "SimulationSystem.h"

#include "cutil.h"
#include "cutil_math.h"
#include "vector_functions.h"

#include <stdio.h>
#include <string.h>

#include "SimulationSystem.h"

#include "SimSnowSPH.cuh"
#include "SimSimpleSPH.cuh"
#include "SimDEM.cuh"

#include "K_Common.inl"

namespace SimLib
{
SimulationSystem::SimulationSystem(bool simpleSph, SimCudaHelper *cudaHelper, bool doKernelTiming)
    : mInitialized(false)
    , mBuffersMapped(false)
    , mCudaTiming(doKernelTiming)
    , mHaveTerrainData(false)
    , hTerrainData(NULL)
    , hTerrainSize(0)
    , hTerrainPosition(make_float3(0))
    , mSimCudaHelper(cudaHelper)
{
    mParticleSimType = simpleSph ? SimulationSimpleSPH : SimulationSnowSPH;
    mSimCudaAllocator = new SimCudaAllocator();
    if (cudaHelper == NULL)
    {
        mSimCudaHelper = new SimCudaHelper();
        mSimCudaHelper->Initialize(0);
    }
};

SimulationSystem::~SimulationSystem()
{
    Free();
};

void SimulationSystem::PrintMemoryUse()
{
    std::cout << "Allocated "
         << mSimCudaAllocator->GetAllocedAmount() << " B / "
         << mSimCudaAllocator->GetAllocedAmount() / 1024.0f / 1024.0f << " MB / "
         << mSimCudaAllocator->GetAllocedAmount() / 1024.0f / 1024.0f / 1024.0f << " GB "
         << "\n";
}

void SimulationSystem::SetExternalBuffer(SimLib::Sim::BaseBufferId id, SimBuffer *buffer)
{
    mExternalSimBuffers[id] = buffer;
}

void SimulationSystem::RemoveExernalBuffer(SimLib::Sim::BaseBufferId id)
{
    mExternalSimBuffers.erase(mExternalSimBuffers.find(id));
}

void SimulationSystem::Init()
{

    mSimulationSteps = 0;

    if (mInitialized)
    {
        Free();
        mInitialized = false;
    }

    switch (mParticleSimType)
    {
    case SimulationSimpleSPH:
    {
        mParticleSim = new Sim::SimpleSPH::SimSimpleSPH(mSimCudaAllocator, mSimCudaHelper);
    }
    break;
    case SimulationSnowSPH:
    {
        mParticleSim = new Sim::SnowSPH::SimSnowSPH(mSimCudaAllocator, mSimCudaHelper);
    }
    break;
        // 		case SimulationDEM:
        // 			{
        // 				Sim::DEM::SimDEM *demSim = new Sim::DEM::SimDEM(mSimCudaAllocator);
        //
        // 				for(std::map<SimLib::Sim::BaseBufferId, SimBuffer*>::const_iterator it = mExternalSimBuffers.begin(); it != mExternalSimBuffers.end(); ++it)
        // 				{
        // 					demSim->SetBuffer(it->first, it->second);
        // 				}
        //
        // 				//TODO configure DEM
        // 				Sim::DEM::DEMParams demParams;
        //
        // 				demParams.scale_to_simulation = 2.0f/volumeSize;
        //
        // 				demParams.particle_radius = 1.0f / 32.0f;
        // 				demParams.collide_dist = 2*demParams.particle_radius;
        // 				demParams.spring = 0.5f;
        // 				demParams.damping = 0.02f;
        // 				demParams.shear = 0.1f;
        // 				demParams.attraction = 0.0f;
        //
        // 				demParams.gravity = make_float3(0.0f, -9.8f, 0.0f);
        //
        // 				demParams.boundary_dampening = 256;
        // 				demParams.boundary_stiffness = 20000;
        // 				demParams.boundary_distance = demParams.collide_dist;
        //
        // 				demSim->SetParams(mNumParticles, volumeSize, demParams);
        // 				particleSim = demSim;
        // 			}
        // 			break;
    }

    for (std::map<SimLib::Sim::BaseBufferId, SimBuffer *>::const_iterator it = mExternalSimBuffers.begin(); it != mExternalSimBuffers.end(); ++it)
    {
        mParticleSim->SetBuffer(it->first, it->second);
    }

    Clear();

    mInitialized = true;
};

void SimulationSystem::SetNumParticles(uint numParticles)
{
    if (mParticleSim == NULL)
        return;

    mParticleSim->GetSettings()->SetValue("Particles Number", numParticles);
}

void SimulationSystem::Free()
{
    delete mParticleSim;
    mParticleSim = NULL;

    unmapRenderingBuffers();
}

void SimulationSystem::mapRenderingBuffers()
{
    // Map the rendering buffers (ready for use by CUDA)
    if (!mBuffersMapped) // && (dParticlesPosVBO != -1 && dParticlesColorVBO != -1))
    {
        bool mapped = true;
        for (std::map<SimLib::Sim::BaseBufferId, SimBuffer *>::const_iterator it = mExternalSimBuffers.begin(); it != mExternalSimBuffers.end(); ++it)
        {
            it->second->MapBuffer();
            mapped &= it->second->IsMapped();
        }
        mBuffersMapped = mapped;
    }
}

void SimulationSystem::unmapRenderingBuffers()
{
    // Unmap the rendering buffers (ready for use by rendering system)
    if (mBuffersMapped) // && (dParticlesPosVBO != -1 && dParticlesColorVBO != -1))
    {
        bool mapped = true;
        for (std::map<SimLib::Sim::BaseBufferId, SimBuffer *>::const_iterator it = mExternalSimBuffers.begin(); it != mExternalSimBuffers.end(); ++it)
        {
            it->second->UnmapBuffer();
            mapped &= !it->second->IsMapped();
        }
        mBuffersMapped = !mapped;
    }
}

void SimulationSystem::Clear()
{
    mapRenderingBuffers();
    mParticleSim->Clear();
    unmapRenderingBuffers();
}

void SimulationSystem::SetTerrainData(float3 terrainPosition, float *terrainHeightData, float4 *terrainNormalData, int terrainSize, float terrainWorldSize)
{
    hTerrainPosition = terrainPosition;
    hTerrainData = terrainHeightData;
    hTerrainSize = terrainSize;
    hTerrainWorldSize = terrainWorldSize;

    dTerrainData.position = terrainPosition;
    dTerrainData.size = terrainSize;
    dTerrainData.world_size = terrainWorldSize;

    CUDA_SAFE_CALL(mSimCudaAllocator->Allocate((void **)&(dTerrainData.heights), dTerrainData.size * dTerrainData.size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(dTerrainData.heights, 0, dTerrainData.size * dTerrainData.size * sizeof(float)));

    CUDA_SAFE_CALL(mSimCudaAllocator->Allocate((void **)&(dTerrainData.normals), dTerrainData.size * dTerrainData.size * sizeof(float4)));
    CUDA_SAFE_CALL(cudaMemset(dTerrainData.normals, 0, dTerrainData.size * dTerrainData.size * sizeof(float4)));

    CUDA_SAFE_CALL(cudaMemcpy(dTerrainData.heights, terrainHeightData, dTerrainData.size * dTerrainData.size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dTerrainData.normals, terrainNormalData, dTerrainData.size * dTerrainData.size * sizeof(float4), cudaMemcpyHostToDevice));

    mHaveTerrainData = true;
}

float SimulationSystem::GetParticleSize()
{
    // meh
    return mParticleSim->GetParticleSize();
}

void SimulationSystem::Simulate(bool mProgress, bool gridWallCollisions)
{
    //if(mSimulationSteps >= 400) return;

    mapRenderingBuffers();

    mParticleSim->Simulate(mCudaTiming, mProgress, gridWallCollisions, mHaveTerrainData, mFluidWorldPosition, dTerrainData);

    //cudaThreadSynchronize();

    unmapRenderingBuffers();

    mSimulationSteps++;
}

void SimulationSystem::SetScene(int scene)
{
    mapRenderingBuffers();

    //ParticleData particleData = mParticleSim->GetParticleData();
    GridParams gridParams = mParticleSim->GetGridParams();

    mParticleSim->Clear();

    int numParticles = (int)mParticleSim->GetSettings()->GetValue("Particles Number");
    float_vec *positions = new float_vec[numParticles];
    memset(positions, 0, numParticles * sizeof(float_vec));

    FillTestData(scene, positions, numParticles, gridParams);

    void *ptr = mParticleSim->GetBuffer(SimLib::Sim::BufferPosition)->GetPtr();

    CUDA_SAFE_CALL(cudaMemcpy(ptr, positions, numParticles * sizeof(float_vec), cudaMemcpyHostToDevice))

    delete[] positions;

    unmapRenderingBuffers();
}

int2 getTerrainPos(float3 const &pos, int const &terrainSize, float const &terrainWorldSize)
{
    int2 terrainPos;
    terrainPos.y = floor(pos.z * (terrainSize / terrainWorldSize));
    terrainPos.x = floor(pos.x * (terrainSize / terrainWorldSize));
    return terrainPos;
}

float getTerrainHeight(int const &terrainPosX, int const &terrainPosZ, float const *terrainHeights, int const &terrainSize)
{
    if (terrainHeights == NULL || terrainSize == 0)
        return 0;
    return terrainHeights[((terrainSize) * (terrainSize)-1) - (((terrainSize)*terrainPosZ)) + terrainPosX];
}

float getTerrainHeight(int2 const &terrainPos, float const *terrainHeights, int const &terrainSize)
{
    return getTerrainHeight(terrainPos.x, terrainPos.y, terrainHeights, terrainSize);
}

void SimulationSystem::SetFluidPosition(float3 fluidWorldPosition)
{
    mFluidWorldPosition = fluidWorldPosition;

    std::cout << "Fluid World Position: " << fluidWorldPosition.x << " " << fluidWorldPosition.y << " " << fluidWorldPosition.z << "\n";
}

void SimulationSystem::FillTestData(int scene, float_vec *position, int numParticles, GridParams hGridParams)
{
    int i = 0;
    float spacing = mParticleSim->GetParticleSpacing();

    switch (scene)
    {
    case 1:
    {
        //cube in corner
        for (float y = hGridParams.grid_min.y; y <= hGridParams.grid_max.y; y += spacing)
        {
            for (float z = hGridParams.grid_min.z / 2.0f; z <= hGridParams.grid_max.z / 1.5f; z += spacing)
            {
                for (float x = hGridParams.grid_min.x / 2.5f; x <= hGridParams.grid_max.x / 2.0f; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }
    }
    break;
    case 2:
    {
        //cube in middle
        for (float y = hGridParams.grid_min.y + hGridParams.grid_size.y / 1.5f; y <= hGridParams.grid_max.y - hGridParams.grid_size.y / 10.0f; y += spacing)
        {
            for (float z = hGridParams.grid_min.z + hGridParams.grid_size.z / 3.5f; z <= hGridParams.grid_max.z - hGridParams.grid_size.z / 2.0f; z += spacing)
            {
                for (float x = hGridParams.grid_min.x + hGridParams.grid_size.x / 3.5f; x <= hGridParams.grid_max.x - hGridParams.grid_size.x / 2.0f; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }
        //equilibrium test
        for (float y = hGridParams.grid_min.y; y <= hGridParams.grid_max.y; y += spacing)
        {
            for (float z = hGridParams.grid_min.z; z <= hGridParams.grid_max.z; z += spacing)
            {
                for (float x = hGridParams.grid_min.x; x <= hGridParams.grid_max.x; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }
    }

    break;
    case 3:
    {
        //cube in middle
        for (float y = hGridParams.grid_min.y + hGridParams.grid_size.y / 2.0f; y <= hGridParams.grid_max.y; y += spacing)
        {
            for (float z = hGridParams.grid_min.z + hGridParams.grid_size.z / 6.0f; z <= hGridParams.grid_max.z - hGridParams.grid_size.z / 6.0f; z += spacing)
            {
                for (float x = hGridParams.grid_min.x + hGridParams.grid_size.x / 6.0f; x <= hGridParams.grid_max.x - hGridParams.grid_size.x / 6.0f; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }

        //equilibrium test
        for (float y = hGridParams.grid_min.y; y <= hGridParams.grid_max.y; y += spacing)
        {
            for (float z = hGridParams.grid_min.z; z <= hGridParams.grid_max.z; z += spacing)
            {
                for (float x = hGridParams.grid_min.x; x <= hGridParams.grid_max.x; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }
    }
    break;
    case 4:
    {
        //small cube in middle
        for (float y = hGridParams.grid_min.y + hGridParams.grid_size.y / 1.5f; y <= hGridParams.grid_max.y - hGridParams.grid_size.y / 10.0f; y += spacing)
        {
            for (float z = hGridParams.grid_min.z + hGridParams.grid_size.z / 3.5f; z <= hGridParams.grid_max.z - hGridParams.grid_size.z / 2.0f; z += spacing)
            {
                for (float x = hGridParams.grid_min.x + hGridParams.grid_size.x / 3.5f; x <= hGridParams.grid_max.x - hGridParams.grid_size.x / 2.0f; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }

        //equilibrium test
        for (float y = hGridParams.grid_min.y; y <= hGridParams.grid_max.y; y += spacing)
        {
            for (float z = hGridParams.grid_min.z; z <= hGridParams.grid_max.z; z += spacing)
            {
                for (float x = hGridParams.grid_min.x; x <= hGridParams.grid_max.x; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }
    }
    break;
    case 5:
    {
        //small cube in middle
        for (float y = hGridParams.grid_min.y + hGridParams.grid_size.y / 1.5f; y <= hGridParams.grid_max.y - hGridParams.grid_size.y / 10.0f; y += spacing)
        {
            for (float z = hGridParams.grid_min.z + hGridParams.grid_size.z / 4.0f; z <= hGridParams.grid_max.z - hGridParams.grid_size.z / 3.0f; z += spacing)
            {
                for (float x = hGridParams.grid_min.x + hGridParams.grid_size.x / 4.0f; x <= hGridParams.grid_max.x - hGridParams.grid_size.x / 3.0f; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }

        //equilibrium test
        for (float y = hGridParams.grid_min.y; y <= hGridParams.grid_max.y; y += spacing)
        {
            for (float z = hGridParams.grid_min.z; z <= hGridParams.grid_max.z; z += spacing)
            {
                for (float x = hGridParams.grid_min.x; x <= hGridParams.grid_max.x; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }
    }
    break;
    case 6:
    {
        //equilibrium test
        for (float y = hGridParams.grid_min.y; y <= hGridParams.grid_max.y; y += spacing)
        {
            for (float z = hGridParams.grid_min.z; z <= hGridParams.grid_max.z; z += spacing)
            {
                for (float x = hGridParams.grid_min.x; x <= hGridParams.grid_max.x; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }
    }
    break;
    case 7:
    {
        //cube in middle
        for (float y = hGridParams.grid_min.y + hGridParams.grid_size.y / 2.0f; y <= hGridParams.grid_max.y; y += spacing)
        {
            for (float z = hGridParams.grid_min.z + hGridParams.grid_size.z / 6.0f; z <= hGridParams.grid_max.z - hGridParams.grid_size.z / 6.0f; z += spacing)
            {
                for (float x = hGridParams.grid_min.x + hGridParams.grid_size.x / 6.0f; x <= hGridParams.grid_max.x - hGridParams.grid_size.x / 6.0f; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }
        break;
    case 8:
    {
        for (int i = 0; i < numParticles; i++)
        {
            position[i] = make_vec(
                (float)rand() / ((float)(RAND_MAX) + 1) * hGridParams.grid_size.x,
                (float)rand() / ((float)(RAND_MAX) + 1) * hGridParams.grid_size.y,
                (float)rand() / ((float)(RAND_MAX) + 1) * hGridParams.grid_size.z);
        }
    }
    break;
    }
    case 9:
    {
        SimSettings *settings = GetSettings();
        float boundaryValue = settings->GetValue("Boundary Distance");

        //small cube in middle
        for (float y = hGridParams.grid_min.y; y <= hGridParams.grid_max.y - hGridParams.grid_size.y / 10.0f; y += spacing)
        {
            for (float z = hGridParams.grid_min.z + hGridParams.grid_size.z / 3.5f; z <= hGridParams.grid_max.z - hGridParams.grid_size.z / 2.0f; z += spacing)
            {
                for (float x = hGridParams.grid_min.x + hGridParams.grid_size.x / 3.5f; x <= hGridParams.grid_max.x - hGridParams.grid_size.x / 2.0f; x += spacing)
                {
                    if (i >= numParticles)
                        break;

                    float3 ppos = make_float3(x, y, z);
                    int2 terrainPos = getTerrainPos(ppos + mFluidWorldPosition + hTerrainPosition, hTerrainSize, hTerrainWorldSize);

                    //is particle above terrain?
                    if (terrainPos.x >= 0 && terrainPos.x < hTerrainSize && terrainPos.y >= 0 && terrainPos.y < hTerrainSize)
                    {
                        // what is the height of the terrain below particle?
                        float terrainHeight = -hTerrainPosition.y - mFluidWorldPosition.y + getTerrainHeight(terrainPos, hTerrainData, hTerrainSize);

                        float boxBottom = hGridParams.grid_min.y + mFluidWorldPosition.y;
                        float terrainBoxBottomDiff = boxBottom - terrainHeight;

                        ppos.y -= terrainBoxBottomDiff;
                        ppos.y += boundaryValue;
                    }

                    position[i] = make_vec(ppos.x, ppos.y, ppos.z, 1);

                    i++;
                }
            }
        }

        //equilibrium test
        for (float y = hGridParams.grid_min.y; y <= hGridParams.grid_max.y; y += spacing)
        {
            for (float z = hGridParams.grid_min.z; z <= hGridParams.grid_max.z; z += spacing)
            {
                for (float x = hGridParams.grid_min.x; x <= hGridParams.grid_max.x; x += spacing)
                {
                    if (i >= numParticles)
                        break;
                    position[i] = make_vec(x, y, z, 1);
                    i++;
                }
            }
        }
    }
    break;
    }

    // safety because we need to use all our particles
    if (i < numParticles)
    {
        for (; i < numParticles; i++)
        {
            position[i] = make_vec(
                (float)rand() / ((float)(RAND_MAX) + 1) * hGridParams.grid_size.x,
                (float)rand() / ((float)(RAND_MAX) + 1) * hGridParams.grid_size.y,
                (float)rand() / ((float)(RAND_MAX) + 1) * hGridParams.grid_size.z);
        }
    }
}
}
