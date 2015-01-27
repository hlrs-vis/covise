#ifndef __SimBase_cuh__
#define __SimBase_cuh__

#include "SimBufferManager.h"
#include "SimBufferCuda.h"
#include "SimSettings.h"

#include "SimCudaHelper.h"
#include "SimCudaAllocator.h"
#include "UniformGrid.cuh"
#include "K_Common.cuh"

#include "timer.h"

namespace SimLib { namespace Sim {

enum BaseBufferId
{
	BufferPosition,
	BufferColor,
	BufferVelocity,
	BufferVeleval,

	BufferPositionSorted,
	BufferColorSorted,
	BufferVelocitySorted,
	BufferVelevalSorted,
};

struct TerrainData
{
	float3  position;
	float*  heights;
	float4* normals;
	int		size;
	float	world_size;
};

struct ParticleData 
{
	// position (in world space)
	float_vec* position;

	// color
	float_vec* color;

	// velocity (in world space)
	float_vec* velocity;

	// vel_eval (in world space, used for leap-frog integration)
	float_vec* veleval;

	//debug
	//float* neighcount;
};

class SimBase : public SettingsChangeCallback
{
public:
	SimBase(SimLib::SimCudaAllocator* SimCudaAllocator, SimLib::SimCudaHelper* simCudaHelper);
	virtual ~SimBase();

	virtual void Clear();

	void SetBuffer(BaseBufferId id, SimLib::SimBuffer* buffer);
	SimBuffer* GetBuffer(BaseBufferId id);
	void RemoveBuffer(BaseBufferId id);

	SimSettings* GetSettings() { return mSettings; };
	GridParams& GetGridParams();

	virtual float GetParticleSize();
	virtual float GetParticleSpacing();
	virtual void Simulate(bool doTiming, bool progress, bool gridWallCollisions, bool terrainCollisions, float3 fluidPosition, TerrainData dTerrainData);

private:
	bool mAlloced;
	bool mParams;

protected:
	virtual void Alloc(uint numParticle);
	virtual void Free();
	virtual void SettingChanged(std::string settingName);


	SimCudaAllocator	*mSimCudaAllocator;
	SimLib::SimCudaHelper	*mSimCudaHelper;

	ocu::GPUTimer	*mGPUTimer;

	uint			mNumParticles;

	// Grid
	UniformGrid		*mUniformGrid;

	BufferManager<BaseBufferId>*	mBaseBuffers;

	// Neighbor list
	NeighborList	hNeighborList;
	NeighborList	dNeighborList;

	SimSettings* mSettings;
};

}} // namespace SimLib { namespace Sim {

#endif