#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "K_Common.inl"
#include "SimSimpleSPH.cuh"

#include "cutil.h"
#include "host_defines.h"
#include "builtin_types.h"

#include "CudaUtils.cuh"
#include "SimulationSystem.h"

#include "timer.h"

namespace SimLib { namespace Sim { namespace SimpleSPH { 

	// Grid textures and constants
#ifdef USE_TEX
	texture<uint, 1, cudaReadModeElementType> neighbors_tex;
	texture<uint, 1, cudaReadModeElementType> cell_indexes_start_tex;
	texture<uint, 1, cudaReadModeElementType> cell_indexes_end_tex; 

	// Fluid textures and constants
	texture<float_vec, 1, cudaReadModeElementType> position_tex;
	texture<float_vec, 1, cudaReadModeElementType> velocity_tex;
	texture<float_vec, 1, cudaReadModeElementType> veleval_tex;
	texture<float, 1, cudaReadModeElementType> pressure_tex;

	texture<float_vec, 1, cudaReadModeElementType> sph_force_tex;
	texture<float_vec, 1, cudaReadModeElementType> color_tex;
	texture<float, 1, cudaReadModeElementType> density_tex;
#endif 


__device__ __constant__	GridParams		cGridParams;
__device__ __constant__	SimpleSPHFluidParams	cFluidParams;
__device__ __constant__	SimpleSPHPrecalcParams	cPrecalcParams;

#include "K_SimpleSPH.inl"
#include "K_UniformGrid_Update.inl"

//#include "cuPrintf.cu"

SimSimpleSPH::SimSimpleSPH(SimLib::SimCudaAllocator* simCudaAllocator, SimLib::SimCudaHelper* simCudaHelper)
: SimBase(simCudaAllocator, simCudaHelper)
	, mSymmetrizationType(SPH_PRESSURE_VISCOPLASTIC)
	, mAlloced(false)
{
	mGPUTimer = new ocu::GPUTimer();

	mSPHBuffers = new SimLib::BufferManager<SimpleSPHBuffers>(mSimCudaAllocator);

	//mSPHBuffers->SetBuffer(BufferSphForce,				new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	//mSPHBuffers->SetBuffer(BufferSphPressure,			new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float)));
	//mSPHBuffers->SetBuffer(BufferSphDensity,			new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float)));
	mSPHBuffers->SetBuffer(BufferSphForceSorted,		new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mSPHBuffers->SetBuffer(BufferSphPressureSorted,		new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float)));
	mSPHBuffers->SetBuffer(BufferSphDensitySorted,		new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float)));

	mSettings->AddSetting("Timestep", 0.002f, 0, 1, "");

	mSettings->AddSetting("Rest Density", 1000, 0, 10000, "kg / m^3");
	mSettings->AddSetting("Rest Pressure", 0, 0, 10000, "");
	mSettings->AddSetting("Ideal Gas Constant", 1, 0.001f, 10, "");
	mSettings->AddSetting("Viscosity", 1, 0, 100, "Pa·s");

	mSettings->AddSetting("Boundary Stiffness", 20000, 0, 100000, "");
	mSettings->AddSetting("Boundary Dampening", 256, 0, 10000, "");
	mSettings->AddSetting("Velocity Limit", 600, 0, 10000, "");
	mSettings->AddSetting("Simulation Scale", 0.001f, 0, 1, "");
	mSettings->AddSetting("Static Friction Limit", 0, 0, 10000, "");
	mSettings->AddSetting("Kinetic Friction", 0, 0, 10000, "");

	mSettings->AddSetting("Particle Mass", ((128*1024.0f)/mSettings->GetValue("Particles Number")) * 0.0002f, 0, 0, "", false);
	mSettings->AddSetting("Particle Rest Distance", 0.87f * pow (mSettings->GetValue("Particle Mass") / mSettings->GetValue("Rest Density"), 1/3.0f ), 0, 0, "", false);
	mSettings->AddSetting("Boundary Distance", 0.5f*mSettings->GetValue("Particle Rest Distance"), 0, 0, "", false);
	mSettings->AddSetting("Smoothing Length", 2*mSettings->GetValue("Particle Rest Distance"), 0, 0, "", false);

	mSettings->SetValue("Grid Cell Size", mSettings->GetValue("Smoothing Length") / mSettings->GetValue("Simulation Scale"));

	//cudaPrintfInit();
}

SimSimpleSPH::~SimSimpleSPH()
{
	Free();
	delete mGPUTimer; mGPUTimer = NULL;
	delete mSPHBuffers; mSPHBuffers = NULL;
}

void SimSimpleSPH::SettingChanged(std::string settingName)
{
	SimBase::SettingChanged(settingName);

	if(settingName == "Particles Number")
	{
		mSettings->SetValue("Particle Mass", (128*1024.0f)/(mSettings->GetValue("Particles Number")) * 0.0002f);
	}
	else if(settingName == "Particle Mass" || settingName == "Rest Density")
	{
		mSettings->SetValue("Particle Rest Distance", 0.87f * pow (mSettings->GetValue("Particle Mass") / mSettings->GetValue("Rest Density"), 1/3.0f ));
	}
	else if(settingName == "Particle Rest Distance")
	{
		mSettings->SetValue("Boundary Distance", 0.5f*mSettings->GetValue("Particle Rest Distance"));
		mSettings->SetValue("Smoothing Length", 2*mSettings->GetValue("Particle Rest Distance"));
	}	
	else if(settingName == "Smoothing Length" || settingName == "Simulation Scale")
	{
		mSettings->SetValue("Grid Cell Size", mSettings->GetValue("Smoothing Length") / mSettings->GetValue("Simulation Scale"));
	}


	UpdateParams();
	//	Alloc(numParticles);

}

void SimSimpleSPH::UpdateParams()
{
	// FLUID SETUP
	hFluidParams.rest_density			= mSettings->GetValue("Rest Density");
	hFluidParams.rest_pressure			= mSettings->GetValue("Rest Pressure");
	hFluidParams.gas_stiffness			= mSettings->GetValue("Ideal Gas Constant");
	hFluidParams.viscosity				= mSettings->GetValue("Viscosity");

	hFluidParams.particle_mass			= mSettings->GetValue("Particle Mass");
	hFluidParams.particle_rest_distance	= mSettings->GetValue("Particle Rest Distance");

	hFluidParams.boundary_distance		= mSettings->GetValue("Boundary Distance");
	hFluidParams.boundary_stiffness		= mSettings->GetValue("Boundary Stiffness");
	hFluidParams.boundary_dampening		= mSettings->GetValue("Boundary Dampening");

	hFluidParams.velocity_limit			= mSettings->GetValue("Velocity Limit");

	hFluidParams.scale_to_simulation	= mSettings->GetValue("Simulation Scale");

	hFluidParams.smoothing_length		= mSettings->GetValue("Smoothing Length");

	hFluidParams.friction_kinetic		= mSettings->GetValue("Kinetic Friction");
	hFluidParams.friction_static_limit	= mSettings->GetValue("Static Friction Limit");

	hPrecalcParams.smoothing_length_pow2 = hFluidParams.smoothing_length * hFluidParams.smoothing_length;
	hPrecalcParams.kernel_poly6_coeff = SPH_Kernels::Wpoly6::Kernel_Constant(hFluidParams.smoothing_length);
	hPrecalcParams.kernel_spiky_grad_coeff = SPH_Kernels::Wspiky::Gradient_Constant(hFluidParams.smoothing_length);
	hPrecalcParams.kernel_viscosity_lap_coeff = SPH_Kernels::Wviscosity::Laplace_Constant(hFluidParams.smoothing_length);

	switch(mSymmetrizationType)
	{
	default:
	case SPH_PRESSURE_MUELLER:
		{
			hPrecalcParams.kernel_pressure_precalc = -0.5f * hPrecalcParams.kernel_spiky_grad_coeff;
		}
		break;
	case SPH_PRESSURE_VISCOPLASTIC:
		{
			hPrecalcParams.kernel_pressure_precalc = -hPrecalcParams.kernel_spiky_grad_coeff;
		}

	}

	hPrecalcParams.kernel_viscosity_precalc = hFluidParams.viscosity * hPrecalcParams.kernel_viscosity_lap_coeff;

	GridParams hGridParams = mUniformGrid->GetGridParams();

	//Copy the grid parameters to the GPU	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cGridParams, &hGridParams, sizeof(GridParams) ) );
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//Copy the fluid parameters to the GPU	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cFluidParams, &hFluidParams, sizeof(SimpleSPHFluidParams) ) );
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//Copy the precalc parameters to the GPU	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cPrecalcParams, &hPrecalcParams, sizeof(SimpleSPHPrecalcParams) ) );
	//CUDA_SAFE_CALL(cudaThreadSynchronize());
}


void SimSimpleSPH::Alloc(uint numParticles)
{
	if (mAlloced)
		return;

	// call base class
	SimBase::Alloc(numParticles);

	mSPHBuffers->AllocBuffers(numParticles);
	
	BindTextures();

	mAlloced = true;
}


void SimSimpleSPH::Free()
{
	SimBase::Free();

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	delete [] hNeighborList.neighbors;
	CUDA_SAFE_CALL(mSimCudaAllocator->Free(dNeighborList.neighbors));
#endif

	UnbindTextures();

	mSPHBuffers->FreeBuffers();

	//cudaPrintfEnd();

	mAlloced = false;
}


void SimSimpleSPH::Clear()
{
	SimBase::Clear();

	mSPHBuffers->MemsetBuffers(0);
}

float SimSimpleSPH::GetParticleSize()
{
	return hFluidParams.particle_rest_distance / hFluidParams.scale_to_simulation;
}

float SimSimpleSPH::GetParticleSpacing()
{
	return hFluidParams.particle_rest_distance / hFluidParams.scale_to_simulation;
}

SimpleSPHFluidParams& SimSimpleSPH::GetFluidParams()
{
	return hFluidParams;
}


void SimSimpleSPH::Simulate(bool doTiming, bool progress, bool gridWallCollisions, bool terrainCollisions, float3 fluidWorldPosition, TerrainData dTerrainData)
{
	float time_hashSPHData,time_radixsort, time_updatelists, time_computeDensity, time_ComputeStep2s, time_integrateForces;

	time_hashSPHData = mUniformGrid->Hash(doTiming, mBaseBuffers->Get(BufferPosition)->GetPtr<float_vec>(), mNumParticles);

	time_radixsort = mUniformGrid->Sort(doTiming);

	time_updatelists = BuildDataStruct(doTiming);

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	time_computeDensity = ComputeDensityAndBuildNeighborList(doTiming);
#else
	time_computeDensity = ComputeStep1(doTiming);
#endif

	time_ComputeStep2s = ComputeStep2(doTiming);

	time_integrateForces = Integrate(doTiming, progress, mSettings->GetValue("Timestep"), gridWallCollisions, terrainCollisions, fluidWorldPosition, dTerrainData);

	if(doTiming)
	{
		char tmp[2048];
		sprintf(tmp,"%4.4f\t%4.4f\t%4.4f\t%4.4f\t%4.4f\t%4.4f\t\n", time_hashSPHData, time_radixsort, time_updatelists, time_computeDensity, time_ComputeStep2s, time_integrateForces);
		printf("%s", tmp);
	}

}

void SimSimpleSPH::BindTextures()
{
	SimpleSPHData dParticleDataSorted = GetParticleDataSorted();

#ifdef USE_TEX
	CUDA_SAFE_CALL(cudaBindTexture(0, position_tex, dParticleDataSorted.position, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velocity_tex, dParticleDataSorted.velocity, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, veleval_tex, dParticleDataSorted.veleval, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, color_tex, dParticleDataSorted.color, mNumParticles*sizeof(float_vec)));

	CUDA_SAFE_CALL(cudaBindTexture(0, sph_force_tex, dParticleDataSorted.sph_force, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, pressure_tex, dParticleDataSorted.pressure, mNumParticles*sizeof(float)));
	CUDA_SAFE_CALL(cudaBindTexture(0, density_tex, dParticleDataSorted.density, mNumParticles*sizeof(float)));

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	CUDA_SAFE_CALL(cudaBindTexture(0, neighbors_tex, dNeighborList.neighbors, dNeighborList.MAX_NEIGHBORS * dNeighborList.numParticles * sizeof(uint)));
#endif

	GridData dGridData = mUniformGrid->GetGridData();
	CUDA_SAFE_CALL(cudaBindTexture(0, cell_indexes_start_tex, dGridData.cell_indexes_start, mUniformGrid->GetNumCells() * sizeof(uint)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cell_indexes_end_tex, dGridData.cell_indexes_end, mUniformGrid->GetNumCells()  * sizeof(uint)));
#endif
}

void SimSimpleSPH::UnbindTextures()
{
#ifdef USE_TEX
	CUDA_SAFE_CALL(cudaUnbindTexture(position_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velocity_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(veleval_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(sph_force_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(color_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(pressure_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(density_tex));

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	CUDA_SAFE_CALL(cudaUnbindTexture(neighbors_tex));
#endif

	CUDA_SAFE_CALL(cudaUnbindTexture(cell_indexes_start_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cell_indexes_end_tex));
#endif
}

SimpleSPHData SimSimpleSPH::GetParticleData()
{
	SimpleSPHData dParticleData;
	dParticleData.color = mBaseBuffers->Get(BufferColor)->GetPtr<float_vec>();
	dParticleData.position = mBaseBuffers->Get(BufferPosition)->GetPtr<float_vec>();
	dParticleData.veleval = mBaseBuffers->Get(BufferVeleval)->GetPtr<float_vec>();
	dParticleData.velocity = mBaseBuffers->Get(BufferVelocity)->GetPtr<float_vec>();
	//dParticleData.sph_force = mSPHBuffers->Get(BufferSphForce)->GetPtr<float_vec>();
	//dParticleData.pressure = mSPHBuffers->Get(BufferSphPressure)->GetPtr<float>();
	//dParticleData.density = mSPHBuffers->Get(BufferSphDensity)->GetPtr<float>();
	return dParticleData;
}

SimpleSPHData SimSimpleSPH::GetParticleDataSorted()
{
	SimpleSPHData dParticleDataSorted;
	dParticleDataSorted.color = mBaseBuffers->Get(BufferColorSorted)->GetPtr<float_vec>();
	dParticleDataSorted.position = mBaseBuffers->Get(BufferPositionSorted)->GetPtr<float_vec>();
	dParticleDataSorted.veleval = mBaseBuffers->Get(BufferVelevalSorted)->GetPtr<float_vec>();
	dParticleDataSorted.velocity = mBaseBuffers->Get(BufferVelocitySorted)->GetPtr<float_vec>();
	dParticleDataSorted.sph_force = mSPHBuffers->Get(BufferSphForceSorted)->GetPtr<float_vec>();
	dParticleDataSorted.pressure = mSPHBuffers->Get(BufferSphPressureSorted)->GetPtr<float>();
	dParticleDataSorted.density = mSPHBuffers->Get(BufferSphDensitySorted)->GetPtr<float>();
	return dParticleDataSorted;
}

float SimSimpleSPH::BuildDataStruct(bool doTiming)
{
	GridData dGridData = mUniformGrid->GetGridData();
	SimpleSPHData dParticleData = GetParticleData();
	SimpleSPHData dParticleDataSorted = GetParticleDataSorted();

	int threadsPerBlock;

	//TODO; this is not correct, need to calculate based on actual device parameters...
	if(mSimCudaHelper->IsFermi())
	{	
		// sm_20:
		// Used 13 registers, 184 bytes cmem[0], 24 bytes cmem[2], 8 bytes cmem[14]
		threadsPerBlock = 256;
	}
	else 
	{
		// sm_13:
		// Used 8 registers, 160+16 bytes smem, 140 bytes cmem[0], 4 bytes cmem[1]
		threadsPerBlock = 256;
	}

	uint numThreads, numBlocks;
	computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);

	while(numBlocks >= 64*1024)
	{
        std::cout << "ALERT: have to rescale threadsPerBlock due to too large grid size >=65536\n";
		threadsPerBlock += 32;
		computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);
	}

	//dynamically allocated shared memory (per block)
	uint smemSize = sizeof(uint)*(numThreads+1);

	// set all cells to empty
	CUDA_SAFE_CALL(cudaMemset(dGridData.cell_indexes_start, 0xff, mUniformGrid->GetNumCells()  * sizeof(uint)));
	
	if(doTiming)
	{
		mGPUTimer->start();
	}

	K_Grid_UpdateSorted<SimpleSPHSystem, SimpleSPHData><<< numBlocks, numThreads, smemSize>>> (
		mNumParticles,
		dParticleData, 
		dParticleDataSorted, 
		dGridData
		);

	//CUT_CHECK_ERROR("Kernel execution failed");

	if(doTiming)
	{
		mGPUTimer->stop();
		return mGPUTimer->elapsed_ms();
	}

	return 0;
}

float SimSimpleSPH::ComputeDensityAndBuildNeighborList(bool doTiming)
{
	GridData dGridData = mUniformGrid->GetGridData();
	SimpleSPHData dParticleData = GetParticleData();
	SimpleSPHData dParticleDataSorted = GetParticleDataSorted();


	//Used 27 registers, 144+16 bytes smem, 156 bytes cmem[0], 8 bytes cmem[1], 8 bytes cmem[14]
	uint threadsPerBlock = 256;

	uint numThreads, numBlocks;
	computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);

	while(numBlocks >= 64*1024)
	{
        std::cout << "ALERT: have to rescale threadsPerBlock due to too large grid size >=65536\n";
		threadsPerBlock += 32;
		computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);
	}

	// set all neighbors to empty
	//CUDA_SAFE_CALL(cudaMemset(dNeighborList.neighbors, 0xff, dNeighborList.MAX_NEIGHBORS * mNumParticles * sizeof(uint)));	
	CUDA_SAFE_CALL(cudaMemset(dNeighborList.neighbors, 0xff, dNeighborList.neighbors_pitch * mNumParticles * sizeof(uint)));	

	if(doTiming)
	{
		mGPUTimer->start();
	}

// 	buildNeighborList<<< numBlocks, numThreads>>> (
// 		mNumParticles,
// 		dNeighborList,
// 		dSPHDataSorted, 
// 		dGridData
// 		);
// 	computeNeighborsAndDensity<<< numBlocks, numThreads>>> (
// 		mNumParticles,
// 		dNeighborList,
// 		dParticleDataSorted,
// 		dGridData
// 		);

	//CUT_CHECK_ERROR("Kernel execution failed");

	if(doTiming)
	{
		mGPUTimer->stop();
		return mGPUTimer->elapsed_ms();
	}

	return 0;
}

float SimSimpleSPH::ComputeStep1(bool doTiming)
{
	GridData dGridData = mUniformGrid->GetGridData();
	SimpleSPHData dParticleData = GetParticleData();
	SimpleSPHData dParticleDataSorted = GetParticleDataSorted();

	uint threadsPerBlock;

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
#ifdef SPHSIMLIB_USE_NEIGHBORLIST_PRECALC_R
	//Used 9 registers, 144+16 bytes smem, 156 bytes cmem[0], 8 bytes cmem[1], 8 bytes cmem[14]
	threadsPerBlock = 128;
#else
	// Used 11 registers, 144+16 bytes smem, 156 bytes cmem[0], 8 bytes cmem[1], 8 bytes cmem[14]
	threadsPerBlock = 128;
#endif
#else
	//TODO; this is not correct, need to calculate based on actual device parameters...
	if(mSimCudaHelper->IsFermi())
	{	
		// sm_2.0 : Used 24 registers, 152 bytes cmem[0], 24 bytes cmem[2], 8 bytes cmem[14]
		threadsPerBlock = 224;
	}
	else 
	{
		// Used 22 registers, 128+16 bytes smem, 140 bytes cmem[0], 4 bytes cmem[1]
		threadsPerBlock = 128;
	}
#endif


	uint numThreads, numBlocks;
	computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);

	while(numBlocks >= 64*1024)
	{
        std::cout << "ALERT: have to rescale threadsPerBlock due to too large grid size >=65536\n";
		threadsPerBlock += 32;
		computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);
	}

	if(doTiming)
	{
		mGPUTimer->start();
	}

	K_SumStep1<<<numBlocks, numThreads>>>(
		mNumParticles,
		dNeighborList,
		dParticleDataSorted,
		dGridData
		);

	//CUT_CHECK_ERROR("Kernel execution failed");

	if(doTiming)
	{
		mGPUTimer->stop();
		return mGPUTimer->elapsed_ms();
	}

	return 0;
}

float SimSimpleSPH::ComputeStep2(bool doTiming)
{
	GridData dGridData = mUniformGrid->GetGridData();
	SimpleSPHData dParticleDataSorted = GetParticleDataSorted();

	uint threadsPerBlock;
#ifdef SPHSIMLIB_USE_NEIGHBORLIST
#ifdef SPHSIMLIB_USE_NEIGHBORLIST_PRECALC_R
	// Used 25 registers, 144+16 bytes smem, 156 bytes cmem[0], 4 bytes cmem[1], 8 bytes cmem[14]
	threadsPerBlock = 320;
#else
	// Used 27 registers, 144+16 bytes smem, 156 bytes cmem[0], 4 bytes cmem[1], 8 bytes cmem[14]
	threadsPerBlock = 64;
#endif
#else
	//TODO; this is not correct, need to calculate based on actual device parameters...
	if(mSimCudaHelper->IsFermi())
	{
		// sm_2.0 (w/32clamp): Used 32 registers, 28+0 bytes lmem, 152 bytes cmem[0], 24 bytes cmem[2], 8 bytes cmem[14]
		threadsPerBlock = 128;
	}
	else 
	{
		// Used 32 registers, 128+16 bytes smem, 140 bytes cmem[0], 4 bytes cmem[1]
		threadsPerBlock = 128;
	}
#endif

	uint numThreads, numBlocks;
	computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);
	
	while(numBlocks >= 64*1024)
	{
        std::cout << "ALERT: have to rescale threadsPerBlock due to too large grid size >=65536\n";
		threadsPerBlock += 32;
		computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);
	}

	if(doTiming)
	{
		mGPUTimer->start();
	}

	switch(mSymmetrizationType)
	{
	default:
	case SPH_PRESSURE_MUELLER:
		{
			K_SumStep2<SPH_PRESSURE_MUELLER><<<numBlocks, numThreads>>>(
				mNumParticles,
				dParticleDataSorted,
#ifdef SPHSIMLIB_USE_NEIGHBORLIST
				dNeighborList,
#else
				dGridData
#endif
				);
		}
		break;
	case SPH_PRESSURE_VISCOPLASTIC:
		{
			K_SumStep2<SPH_PRESSURE_VISCOPLASTIC> <<<numBlocks, numThreads>>>(
				mNumParticles,
				dParticleDataSorted,
#ifdef SPHSIMLIB_USE_NEIGHBORLIST
				dNeighborList,
#else
				dGridData
#endif
				);
		}
		break;
	}

	//CUT_CHECK_ERROR("Kernel execution failed");
	if(doTiming)
	{
		mGPUTimer->stop();
		return mGPUTimer->elapsed_ms();
	}

	return 0;
}

float SimSimpleSPH::Integrate(bool doTiming, bool progress, float deltaTime, bool gridWallCollisions, bool terrainCollisions, float3 fluidWorldPosition, TerrainData dTerrainData)
{
	GridData		dGridData	= mUniformGrid->GetGridData();
	SimpleSPHData	dParticleData = GetParticleData();
	SimpleSPHData	dParticleDataSorted = GetParticleDataSorted();

	int threadsPerBlock;

	//TODO; this is not correct, need to calculate based on actual device parameters...
	if(mSimCudaHelper->IsFermi())
	{
		//sm_20: Used 26 registers, 248 bytes cmem[0], 24 bytes cmem[2], 8 bytes cmem[14], 28 bytes cmem[16]
		threadsPerBlock = 416;
	}
	else 
	{
		//sm_13: Used 23 registers, 224+16 bytes smem, 140 bytes cmem[0], 40 bytes cmem[1]
		threadsPerBlock = 192;
	}

	//Used 25 registers, 208+16 bytes smem, 144 bytes cmem[0], 16 bytes cmem[1]
	uint numThreads, numBlocks;
	computeGridSize(mNumParticles, threadsPerBlock, numBlocks, numThreads);


	if(doTiming)
	{
		mGPUTimer->start();
	}

	K_Integrate<Velocity, HSVBlueToRed><<<numBlocks, numThreads>>>(
		mNumParticles,
		gridWallCollisions, terrainCollisions,
		deltaTime,
		progress,
		dGridData,
		dParticleData,
		dParticleDataSorted,
		fluidWorldPosition,
		dTerrainData
		);

	//CUT_CHECK_ERROR("Kernel execution failed");


	//cudaPrintfDisplay(stdout, true);

	if(doTiming)
	{
		mGPUTimer->stop();
		return mGPUTimer->elapsed_ms();
	}

	return 0;
}

}}} // namespace SimLib { namespace Sim { namespace SimpleSPH { 
