#ifndef __DEM_cuh__
#define __DEM_cuh__

#include "SimCudaAllocator.h"

#include "SimBase.cuh"
#include "UniformGrid.cuh"

#include "K_Common.cuh"
#include "timer.h"

typedef unsigned int uint;

namespace SimLib { namespace Sim { namespace DEM { 

enum DEMBufferID
{
	BufferForce,
	BufferForceSorted
};

struct DEMParams 
{
	float particle_radius;

	float spring;
	float damping;
	float shear;
	float attraction;

	float  collide_dist;    
	float3 gravity;


	// the scale of the simulation (ie. 0.1 of world scale)
	float			scale_to_simulation;

	// external stiffness (against boundaries)
	float			boundary_stiffness; 

	// external dampening (against boundaries)
	float			boundary_dampening;

	// the distance a particle will be "pushed" away from boundaries
	float			boundary_distance;
};

struct DEMData : ParticleData
{
	float_vec *force;
};

class SimDEM : public SimBase
{
public:
	SimDEM(SimLib::SimCudaAllocator* SimCudaAllocator);
	~SimDEM();

	void Clear();

	void SetParams(uint numParticles, float gridWorldSize, DEMParams &fluidParams);
	void Alloc(uint numParticles);

	void Simulate(bool doTiming, bool progress, bool gridWallCollisions, bool terrainCollisions, TerrainData dTerrainData);

	float GetParticleSize();
	float GetParticleSpacing();

	DEMParams& GetFluidParams();
	DEMData GetParticleDataSorted();
	DEMData GetParticleData();

private:
	bool mAlloced;
	bool mParams;

	ocu::GPUTimer *mGPUTimer;

protected:
	void Free();

	float BuildDataStruct(bool doTiming);
	float BuildNeighborList(bool doTiming);
	float ComputeCollisions(bool doTiming);
	float Integrate(bool doTiming, bool progress, float deltaTime, bool gridWallCollisions, bool terrainCollisions, TerrainData dTerrainData);

	void BindTextures();
	void UnbindTextures();

	 BufferManager<DEMBufferID> *mDEMBuffers;

	// params
	DEMParams hDEMParams;
};

}}} // namespace SimLib { namespace Sim { namespace SimpleSPH { 

#endif