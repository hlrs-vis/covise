#ifndef __SimBase_cu__
#define __SimBase_cu__

#include "SimBase.cuh"

using namespace SimLib;

namespace SimLib { namespace Sim { 
	
__global__ void FermiCacheOverride()
{
}

SimBase::SimBase(SimCudaAllocator* simCudaAllocator, SimLib::SimCudaHelper* simCudaHelper)
	: mSimCudaAllocator(simCudaAllocator)
	, mSimCudaHelper(simCudaHelper)
	, mAlloced(false)
	, mNumParticles(0)
{
	mGPUTimer = new ocu::GPUTimer();

	mBaseBuffers = new BufferManager<BaseBufferId>(mSimCudaAllocator);
	mBaseBuffers->SetBuffer(BufferPosition,			new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mBaseBuffers->SetBuffer(BufferVelocity,			new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mBaseBuffers->SetBuffer(BufferVeleval,			new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mBaseBuffers->SetBuffer(BufferColor,			new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mBaseBuffers->SetBuffer(BufferPositionSorted,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mBaseBuffers->SetBuffer(BufferVelocitySorted,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mBaseBuffers->SetBuffer(BufferVelevalSorted,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mBaseBuffers->SetBuffer(BufferColorSorted,		new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));

	mUniformGrid = new UniformGrid(mSimCudaAllocator, mSimCudaHelper);

	mSettings = new SimSettings();
	mSettings->AddCallback(this);

	mSettings->AddSetting("Particles Number", 16384, 0, 0, "");
	mSettings->AddSetting("Grid World Size", 256, 0, 0, "World Units");
	mSettings->AddSetting("Grid Cell Size", 10, 0, 0, "World Units");

	//TODO; this is not correct, need to calculate based on actual device parameters...
	if(mSimCudaHelper->IsFermi())
	{
		cudaFuncSetCacheConfig(FermiCacheOverride, cudaFuncCachePreferL1);
	}
	else 
	{
	}

}

SimBase::~SimBase()
{
	Free();

	delete mGPUTimer; mGPUTimer = NULL;
	delete mUniformGrid; mUniformGrid = NULL;
	delete mBaseBuffers; mBaseBuffers = NULL;
}

void SimBase::SettingChanged(std::string settingName)
{
	if(settingName == "Particles Number" || settingName == "Grid World Size" || settingName == "Grid Cell Size")
	{
		mNumParticles = mSettings->GetValue("Particles Number");
		Free();
		Alloc(mNumParticles);
	}
}

void SimBase::Alloc(uint numParticles)
{
// 	if(!mParams)
// 	{
// 		printf("SimBase::Alloc, no params!");
// 		return;
// 	}	

	if (mAlloced)
		return;

	mNumParticles = numParticles;

	mUniformGrid->Alloc(numParticles, mSettings->GetValue("Grid Cell Size"), mSettings->GetValue("Grid World Size"));

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	hNeighborList.MAX_NEIGHBORS = 100;
	hNeighborList.numParticles = dParticleData.numParticles;
	dNeighborList.MAX_NEIGHBORS = hNeighborList.MAX_NEIGHBORS;
	dNeighborList.numParticles = dParticleData.numParticles;

	hNeighborList.neighbors = new uint[hNeighborList.MAX_NEIGHBORS * hNeighborList.numParticles];
	memset(hNeighborList.neighbors, 0 , hNeighborList.MAX_NEIGHBORS * hNeighborList.numParticles * sizeof(uint));

	CUDA_SAFE_CALL(un->Allocate((void**) &(dNeighborList.neighbors), dNeighborList.MAX_NEIGHBORS * dNeighborList.numParticles * sizeof(uint)));
	dNeighborList.neighbors_pitch = dNeighborList.MAX_NEIGHBORS;

	//size_t pitchInBytes;
	//CUDA_SAFE_CALL(mSimCudaAllocator->AllocatePitch((void**) &(dNeighborList.neighbors), &pitchInBytes,	dNeighborList.MAX_NEIGHBORS * sizeof(uint), dNeighborList.numParticles));
	// want pitch in elements, not bytes
	//dNeighborList.neighbors_pitch = (int)pitchInBytes / sizeof(uint);
#endif

	// Allocate device particle buffers
	mBaseBuffers->AllocBuffers(numParticles);

	mAlloced = true;
}

void SimBase::Free()
{
	if(!mAlloced) return;

	// free grid
	mUniformGrid->Free();

	// Free allocated GPU memory
	mBaseBuffers->FreeBuffers();
	
	mAlloced = false;
}

void SimBase::Clear()
{
	mUniformGrid->Clear();

	mBaseBuffers->MemsetBuffers(0);
}


void SimBase::SetBuffer(BaseBufferId id, SimBuffer* buffer)
{
	mBaseBuffers->SetBuffer(id, buffer);
}

void SimBase::RemoveBuffer(BaseBufferId id)
{
	mBaseBuffers->RemoveBuffer(id);
}

GridParams& SimBase::GetGridParams()
{
	return mUniformGrid->GetGridParams();
}

float SimBase::GetParticleSize()
{
	return 1;
}
float SimBase::GetParticleSpacing()
{
	return 2;
}

SimBuffer* SimBase::GetBuffer(BaseBufferId id)
{
	return mBaseBuffers->Get(id);
}

void SimBase::Simulate(bool doTiming, bool progress, bool gridWallCollisions, bool terrainCollisions, float3 fluidWorldPosition, TerrainData dTerrainData)
{

}

}} // namespace SimLib { namespace Sim { 
#endif