// #include <stdlib.h>
// #include <stdio.h>
// #include <math.h>
// #include <string.h>
// 
// //#include "cuPrintf.cu"
// #include "K_Common.cuh"
// #include "cutil.h"
// #include "host_defines.h"
// #include "builtin_types.h"
// 
// #include "SimDEM.cuh"
// #include "CudaUtils.cuh"
// 
// 
// // Grid textures and constants
// #ifdef USE_TEX
// texture<uint, 1, cudaReadModeElementType> neighbors_tex;
// texture<uint, 1, cudaReadModeElementType> cell_indexes_start_tex;
// texture<uint, 1, cudaReadModeElementType> cell_indexes_end_tex; 
// 
// // Fluid textures and constants
// texture<float_vec, 1, cudaReadModeElementType> position_tex;
// texture<float_vec, 1, cudaReadModeElementType> velocity_tex;
// texture<float_vec, 1, cudaReadModeElementType> veleval_tex;
// texture<float_vec, 1, cudaReadModeElementType> color_tex;
// texture<float_vec, 1, cudaReadModeElementType> force_tex;
// #endif 
// 
// namespace SimLib { namespace Sim { namespace DEM { 
// 
// __device__ __constant__	DEMParams		cDEMParams;
// __device__ __constant__	GridParams		cGridParams;
// 
// #include "K_SimDEM.inl"
// 
// SimDEM::SimDEM(SimCudaAllocator* SimCudaAllocator)
// : SimBase(SimCudaAllocator)
// , mAlloced(false)
// {
// 	mGPUTimer = new ocu::GPUTimer();
// 
// 	mDEMBuffers = new BufferManager<DEMBufferID>();
// 
// 	mDEMBuffers->SetBuffer(BufferForce,			new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
// 	mDEMBuffers->SetBuffer(BufferForceSorted,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
// }
// 
// SimDEM::~SimDEM()
// {
// 	delete mGPUTimer; mGPUTimer = NULL;
// 	delete mDEMBuffers; mDEMBuffers = NULL;
// }
// 
// 
// void SimDEM::SetParams(uint numParticles, float gridWorldSize, DEMParams &demParams)
// {
// 	hDEMParams = demParams;
// 
// 	// call base class
// 	SimBase::SetParams(demParams.collide_dist/demParams.scale_to_simulation, gridWorldSize);
// 
// 	Alloc(numParticles);
// 
// 	GridParams hGridParams = mUniformGrid->GetGridParams();
// 
// 	//Copy the grid parameters to the GPU	
// 	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cGridParams, &hGridParams, sizeof(GridParams) ) );
// 	CUDA_SAFE_CALL(cudaThreadSynchronize());
// 
// 	//Copy the fluid parameters to the GPU	
// 	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cDEMParams, &hDEMParams, sizeof(DEMParams) ) );
// 	CUDA_SAFE_CALL(cudaThreadSynchronize());
// 
// }
// 
// void SimDEM::Alloc(uint numParticles)
// {
// 	if(!mParams)
// 	{
// 		printf("SimDEM::Alloc, no params!");
// 		return;
// 	}	
// 
// 	if (mAlloced)
// 		Free();
// 
// 	// call base class
// 	SimBase::Alloc(numParticles);
// 
// 	mNumParticles = numParticles;
// 
// 	mDEMBuffers->AllocBuffers(mNumParticles);
// 
// //	cudaPrintfInit();
// 
// 	BindTextures();
// 
// 	mAlloced = true;
// }
// 
// 
// void SimDEM::Free()
// {
// 	SimBase::Free();
// 
// 	UnbindTextures();
// 
// 	mDEMBuffers->FreeBuffers();
// 
// //	cudaPrintfEnd();
// 
// 	mAlloced = false;
// }
// 
// 
// void SimDEM::Clear()
// {
// 	SimBase::Clear();
// 
// 	mDEMBuffers->MemsetBuffers(0);
// }
// 
// DEMParams& SimDEM::GetFluidParams()
// {
// 	return hDEMParams;
// }
// 
// float SimDEM::GetParticleSize()
// {
// 	return hDEMParams.particle_radius/hDEMParams.scale_to_simulation;
// }
// float SimDEM::GetParticleSpacing()
// {
// 	return 2*hDEMParams.particle_radius/hDEMParams.scale_to_simulation;
// }
// 
// void SimDEM::Simulate(bool doTiming, bool progress, bool gridWallCollisions, bool terrainCollisions, TerrainData dTerrainData)
// {
// 	float time_hash,time_radixsort, time_updatelists, time_computeCollisions, time_integrateForces;
// 
// 	time_hash = mUniformGrid->Hash(doTiming, mBaseBuffers->Get(BufferPosition)->GetPtr<float_vec>(), mNumParticles);
// 
// 	time_radixsort = mUniformGrid->Sort(doTiming);
// 
// 	time_updatelists = BuildDataStruct(doTiming);
// 
// 	time_computeCollisions = ComputeCollisions(doTiming);
// 
// 	time_integrateForces = Integrate(doTiming, progress,  mSettings->GetValue("Timestep"), gridWallCollisions, terrainCollisions, dTerrainData);
// 
// 	if(doTiming)
// 	{
// 		char tmp[2048];
// 		sprintf(tmp,"%4.4f\t%4.4f\t%4.4f\t%4.4f\t%4.4f\t\n", time_hash, time_radixsort, time_updatelists,  time_computeCollisions, time_integrateForces);
// 		printf(tmp);
// 	}
// }
// 
// void SimDEM::BindTextures()
// {
// 	DEMData dParticleDataSorted = GetParticleDataSorted();
// 
// #ifdef USE_TEX
// 	CUDA_SAFE_CALL(cudaBindTexture(0, position_tex, dParticleDataSorted.position, mNumParticles*sizeof(float_vec)));
// 	CUDA_SAFE_CALL(cudaBindTexture(0, velocity_tex, dParticleDataSorted.velocity, mNumParticles*sizeof(float_vec)));
// 	CUDA_SAFE_CALL(cudaBindTexture(0, veleval_tex, dParticleDataSorted.veleval, mNumParticles*sizeof(float_vec)));
// 	CUDA_SAFE_CALL(cudaBindTexture(0, color_tex, dParticleDataSorted.color, mNumParticles*sizeof(float_vec)));
// 	CUDA_SAFE_CALL(cudaBindTexture(0, force_tex, dParticleDataSorted.force, mNumParticles*sizeof(float_vec)));
// 
// #ifdef SPHSIMLIB_USE_NEIGHBORLIST
// 	CUDA_SAFE_CALL(cudaBindTexture(0, neighbors_tex, dNeighborList.neighbors, dNeighborList.MAX_NEIGHBORS * dNeighborList.numParticles * sizeof(uint)));
// #endif
// 
// 	GridData dGridData = mUniformGrid->GetGridData();
// 	CUDA_SAFE_CALL(cudaBindTexture(0, cell_indexes_start_tex, dGridData.cell_indexes_start, mUniformGrid->GetNumCells()  * sizeof(uint)));
// 	CUDA_SAFE_CALL(cudaBindTexture(0, cell_indexes_end_tex, dGridData.cell_indexes_end, mUniformGrid->GetNumCells()  * sizeof(uint)));
// #endif
// }
// 
// void SimDEM::UnbindTextures()
// {
// #ifdef USE_TEX
// 	CUDA_SAFE_CALL(cudaUnbindTexture(position_tex));
// 	CUDA_SAFE_CALL(cudaUnbindTexture(velocity_tex));
// 	CUDA_SAFE_CALL(cudaUnbindTexture(veleval_tex));
// 	CUDA_SAFE_CALL(cudaUnbindTexture(color_tex));
// 	CUDA_SAFE_CALL(cudaUnbindTexture(force_tex));
// 
// #ifdef SPHSIMLIB_USE_NEIGHBORLIST
// 	CUDA_SAFE_CALL(cudaUnbindTexture(neighbors_tex));
// #endif
// 
// 	GridData dGridData = mUniformGrid->GetGridData();
// 	CUDA_SAFE_CALL(cudaUnbindTexture(cell_indexes_start_tex));
// 	CUDA_SAFE_CALL(cudaUnbindTexture(cell_indexes_end_tex));
// #endif
// }
// 
// DEMData SimDEM::GetParticleDataSorted()
// {
// 	DEMData dParticleDataSorted;
// 	dParticleDataSorted.color = mBaseBuffers->Get(BufferColorSorted)->GetPtr<float_vec>();
// 	dParticleDataSorted.position = mBaseBuffers->Get(BufferPositionSorted)->GetPtr<float_vec>();
// 	dParticleDataSorted.veleval = mBaseBuffers->Get(BufferVelevalSorted)->GetPtr<float_vec>();
// 	dParticleDataSorted.velocity = mBaseBuffers->Get(BufferVelocitySorted)->GetPtr<float_vec>();
// 	dParticleDataSorted.force = mDEMBuffers->Get(BufferForceSorted)->GetPtr<float_vec>();
// 	return dParticleDataSorted;
// }
// 
// DEMData SimDEM::GetParticleData()
// {
// 	DEMData dParticleData;
// 	dParticleData.color = mBaseBuffers->Get(BufferColor)->GetPtr<float_vec>();
// 	dParticleData.position = mBaseBuffers->Get(BufferPosition)->GetPtr<float_vec>();
// 	dParticleData.veleval = mBaseBuffers->Get(BufferVeleval)->GetPtr<float_vec>();
// 	dParticleData.velocity = mBaseBuffers->Get(BufferVelocity)->GetPtr<float_vec>();
// 	dParticleData.force = mDEMBuffers->Get(BufferForce)->GetPtr<float_vec>();
// 
// 	return dParticleData;
// }
// 
// 
// float SimDEM::BuildDataStruct(bool doTiming)
// {
// 	GridData dGridData = mUniformGrid->GetGridData();
// 	DEMData dParticleData = GetParticleData();
// 	DEMData dParticleDataSorted = GetParticleDataSorted();
// 
// 	// Used 10 registers, 192+16 bytes smem, 144 bytes cmem[0], 12 bytes cmem[1]
// 	uint numThreads, numBlocks;
// 	computeGridSize(mNumParticles, 128, numBlocks, numThreads);
// 
// 	//dynamically allocated shared memory (per block)
// 	uint smemSize = sizeof(uint)*(numThreads+1);
// 
// 	if(doTiming)
// 	{
// 		mGPUTimer->start();
// 	}
// 
// 	// set all cells to empty
// 	CUDA_SAFE_CALL(cudaMemset(dGridData.cell_indexes_start, 0xff, mUniformGrid->GetNumCells()  * sizeof(uint)));
// 
// 	K_Grid_UpdateSorted<DEMSystem, DEMData><<< numBlocks, numThreads, smemSize>>> (
// 		mNumParticles,
// 		dParticleData, 
// 		dParticleDataSorted, 
// 		dGridData
// 		);
// 
// 	//CUT_CHECK_ERROR("Kernel execution failed");
// 
// 	if(doTiming)
// 	{
// 		mGPUTimer->stop();
// 		return mGPUTimer->elapsed_ms();
// 	}
// 
// 	return 0;
// }
// 
// 
// float SimDEM::ComputeCollisions(bool doTiming)
// {
// 	GridData dGridData = mUniformGrid->GetGridData();
// 	DEMData dParticleDataSorted = GetParticleDataSorted();
// 
// 	// Used 25 registers, 144+16 bytes smem, 160 bytes cmem[0], 8 bytes cmem[1], 8 bytes cmem[14]
// 	uint threadsPerBlock = 320;
// 
// 	uint numThreads, numBlocks;
// 	computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);
// 
// 	if(doTiming)
// 	{
// 		mGPUTimer->start();
// 	}
// 
// 	computeCollisions<<<numBlocks, numThreads>>>(
// 		mNumParticles,
// 		dNeighborList,
// 		dParticleDataSorted,
// 		dGridData
// 		);
// 
// 	//CUT_CHECK_ERROR("Kernel execution failed");
// 
// 	if(doTiming)
// 	{
// 		mGPUTimer->stop();
// 		return mGPUTimer->elapsed_ms();
// 	}
// 
// 	return 0;
// }
// 
// float SimDEM::Integrate(bool doTiming, bool progress, float deltaTime, bool gridWallCollisions, bool terrainCollisions, TerrainData dTerrainData)
// {
// 	GridData	dGridData				= mUniformGrid->GetGridData();
// 	DEMData		dParticleData			= GetParticleData();
// 	DEMData		dParticleDataSorted		= GetParticleDataSorted();
// 
// 	//Used 25 registers, 208+16 bytes smem, 144 bytes cmem[0], 16 bytes cmem[1]
// 	uint numThreads, numBlocks;
// 	computeGridSize(mNumParticles, 320, numBlocks, numThreads);
// 
// 	if(doTiming)
// 	{
// 		mGPUTimer->start();
// 	}
// 
// 	integrateDEM<<<numBlocks, numThreads>>>(
// 		mNumParticles,
// 		gridWallCollisions, terrainCollisions,
// 		deltaTime,
// 		progress,
// 		dGridData,
// 		dParticleData,
// 		dParticleDataSorted,
// 		dTerrainData
// 		);
// 
// 	//CUT_CHECK_ERROR("Kernel execution failed");
// 
// 	//cudaPrintfDisplay(stdout, true);
// 
// 	if(doTiming)
// 	{
// 		mGPUTimer->stop();
// 		return mGPUTimer->elapsed_ms();
// 	}
// 
// 	return 0;
// }
// 
// }}} // namespace SimLib { namespace Sim { namespace SimpleSPH { 