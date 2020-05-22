#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>

#include "K_Common.inl"
#include "SimSnowSPH.cuh"

#include "cutil.h"
#include "host_defines.h"
#include "builtin_types.h"

#include "CudaUtils.cuh"
#include "SimulationSystem.h"

#include "timer.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>

#ifdef USE_TEX
// Grid textures and constants
texture<uint, 1, cudaReadModeElementType> neighbors_tex;
texture<uint, 1, cudaReadModeElementType> cell_indexes_start_tex;
texture<uint, 1, cudaReadModeElementType> cell_indexes_end_tex; 

// Fluid textures and constants
texture<float_vec, 1, cudaReadModeElementType> position_tex;
texture<float_vec, 1, cudaReadModeElementType> color_tex;
texture<float_vec, 1, cudaReadModeElementType> velocity_tex;
texture<float_vec, 1, cudaReadModeElementType> veleval_tex;

texture<float_vec, 1, cudaReadModeElementType> sph_force_tex;
texture<float_vec, 1, cudaReadModeElementType> xsph_tex;
texture<float, 1, cudaReadModeElementType> density_tex;
texture<float, 1, cudaReadModeElementType> pressure_tex;
texture<float_vec, 1, cudaReadModeElementType> stress_tensor_tex;
#endif 

namespace SimLib { namespace Sim { namespace SnowSPH { 

__device__ __constant__	GridParams				cGridParams;
__device__ __constant__	SnowSPHFluidParams		cFluidParams;
__device__ __constant__	SnowSPHPrecalcParams	cPrecalcParams;

//#include "cuPrintf.cu"
#include "K_SnowSPH.inl"
#include "K_UniformGrid_Update.inl"

SimSnowSPH::SimSnowSPH(SimLib::SimCudaAllocator* simCudaAllocator, SimLib::SimCudaHelper* simCudaHelper)
: SimBase(simCudaAllocator, simCudaHelper)
, mAlloced(false)
{
	mSPHBuffers = new SimLib::BufferManager<SnowSPHBuffers>(mSimCudaAllocator);

	mSPHBuffers->SetBuffer(BufferXSPHSorted,			new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mSPHBuffers->SetBuffer(BufferSphForceSorted,		new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float_vec)));
	mSPHBuffers->SetBuffer(BufferSphPressureSorted,		new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float)));
	mSPHBuffers->SetBuffer(BufferSphDensitySorted,		new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float)));
	mSPHBuffers->SetBuffer(BufferSphStressTensorSorted,	new SimBufferCuda(mSimCudaAllocator, Device, sizeof(matrix3)));
	//mSPHBuffers->SetBuffer(BufferCFLSorted,				new SimBufferCuda(mSimCudaAllocator, Device, sizeof(float)));


	mSettings->AddSetting("Particles Number", 32*1024, 1024, 0, "");
	mSettings->AddSetting("Grid World Size", 1024, 1, 0, "");

	mSettings->AddSetting("Timestep", 0.0005f, 0, 1, "");

	mSettings->AddSetting("Rest Density", 1000, 0, 10000, "kg / m^3");
	mSettings->AddSetting("Rest Pressure", 0, 0, 10000, "");
	mSettings->AddSetting("Ideal Gas Constant", 1.5f, 0.001f, 10, "");
	mSettings->AddSetting("Viscosity", 1, 0, 100, "Pa·s");

 	mSettings->AddSetting("Boundary Stiffness", 20000, 0, 100000, "");
 	mSettings->AddSetting("Boundary Dampening", 256, 0, 10000, "");
 	mSettings->AddSetting("Velocity Limit", 600, 0, 10000, "");
 	mSettings->AddSetting("Simulation Scale", 0.001f, 0, 1, "");
	mSettings->AddSetting("XSPH Factor", 0.5, 0, 1, "");
	mSettings->AddSetting("Static Friction Limit", 0, 0, 10000, "");
	mSettings->AddSetting("Kinetic Friction", 0, 0, 10000, "");

	mSettings->AddSetting("Particle Mass", ((128*1024.0f)/mSettings->GetValue("Particles Number")) * 0.0002f, 0, 0, "", false);
	mSettings->AddSetting("Particle Rest Distance", 0.87f * pow (mSettings->GetValue("Particle Mass") / mSettings->GetValue("Rest Density"), 1/3.0f ), 0, 0, "", false);
	mSettings->AddSetting("Boundary Distance", 0.5f*mSettings->GetValue("Particle Rest Distance"), 0, 0, "", false);
	mSettings->AddSetting("Smoothing Length", 2*mSettings->GetValue("Particle Rest Distance"), 0, 0, "", false);

	mSettings->SetValue("Grid Cell Size", mSettings->GetValue("Smoothing Length") / mSettings->GetValue("Simulation Scale"));

	//cudaPrintfInit();
}

SimSnowSPH::~SimSnowSPH()
{
	Free();
	delete mSPHBuffers; mSPHBuffers = NULL;
}

void SimSnowSPH::SettingChanged(std::string settingName)
{
	SimBase::SettingChanged(settingName);

	if(settingName == "Particles Number")
	{ 
		//ensure multiple of block size!!
		int numParticles = (int)mSettings->GetValue("Particles Number");
		mSettings->SetValue("Particles Number", (float)(numParticles + ((numParticles % 256))));

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

void SimSnowSPH::UpdateParams()
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

	hFluidParams.xsph_factor			= mSettings->GetValue("XSPH Factor");

	hFluidParams.friction_static_limit		= mSettings->GetValue("Static Friction Limit");
	hFluidParams.friction_kinetic		= mSettings->GetValue("Kinetic Friction");

	hPrecalcParams.smoothing_length_pow2 = hFluidParams.smoothing_length * hFluidParams.smoothing_length;
	hPrecalcParams.smoothing_length_pow3 = hFluidParams.smoothing_length * hFluidParams.smoothing_length * hFluidParams.smoothing_length;
	hPrecalcParams.smoothing_length_pow4 = hFluidParams.smoothing_length * hFluidParams.smoothing_length * hFluidParams.smoothing_length * hFluidParams.smoothing_length;
	hPrecalcParams.smoothing_length_pow5 = hFluidParams.smoothing_length * hFluidParams.smoothing_length * hFluidParams.smoothing_length * hFluidParams.smoothing_length * hFluidParams.smoothing_length;

	hPrecalcParams.kernel_poly6_coeff = SPH_Kernels::Wpoly6::Kernel_Constant(hFluidParams.smoothing_length);
	hPrecalcParams.kernel_spiky_grad_coeff = SPH_Kernels::Wspiky::Gradient_Constant(hFluidParams.smoothing_length);
	hPrecalcParams.kernel_viscosity_lap_coeff = SPH_Kernels::Wviscosity::Laplace_Constant(hFluidParams.smoothing_length);
	hPrecalcParams.kernel_pressure_precalc = -hPrecalcParams.kernel_spiky_grad_coeff;

	hPrecalcParams.kernel_viscosity_precalc = hFluidParams.viscosity * hPrecalcParams.kernel_viscosity_lap_coeff;

	GridParams hGridParams = mUniformGrid->GetGridParams();

	//Copy the grid parameters to the GPU	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cGridParams, &hGridParams, sizeof(GridParams) ) );
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//Copy the fluid parameters to the GPU	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cFluidParams, &hFluidParams, sizeof(SnowSPHFluidParams) ) );
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//Copy the precalc parameters to the GPU	
	CUDA_SAFE_CALL(cudaMemcpyToSymbol (cPrecalcParams, &hPrecalcParams, sizeof(SnowSPHPrecalcParams) ) );
	//CUDA_SAFE_CALL(cudaThreadSynchronize());
}


void SimSnowSPH::Alloc(uint numParticles)
{
	if (mAlloced)
		return;

	// call base class
	SimBase::Alloc(numParticles);

	mSPHBuffers->AllocBuffers(numParticles);

	BindTextures();

	mAlloced = true;
}


void SimSnowSPH::Free()
{
	SimBase::Free();

	if(!mAlloced) return;

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	delete [] hNeighborList.neighbors;
	CUDA_SAFE_CALL(mSimCudaAllocator->Free(dNeighborList.neighbors));
#endif

	UnbindTextures();

	mSPHBuffers->FreeBuffers();

	//cudaPrintfEnd();

	mAlloced = false;
}


void SimSnowSPH::Clear()
{
	SimBase::Clear();

	mSPHBuffers->MemsetBuffers(0);
}

float SimSnowSPH::GetParticleSize()
{
	return 0.7f*hFluidParams.smoothing_length / hFluidParams.scale_to_simulation;
}

float SimSnowSPH::GetParticleSpacing()
{
	return hFluidParams.particle_rest_distance / hFluidParams.scale_to_simulation;
}

SnowSPHFluidParams& SimSnowSPH::GetFluidParams()
{
	return hFluidParams;
}

void SimSnowSPH::Simulate(bool doTiming, bool progress, bool gridWallCollisions, bool terrainCollisions, float3 fluidWorldPosition, TerrainData dTerrainData)
{

	float time_hashSPHData,time_radixsort, time_updatelists, time_computeStep1, time_ComputeStep2, time_ComputeStep3, time_integrateForces;

	time_hashSPHData = mUniformGrid->Hash(doTiming, mBaseBuffers->Get(BufferPosition)->GetPtr<float_vec>(), mNumParticles);

	time_radixsort = mUniformGrid->Sort(doTiming);

	time_updatelists = BuildDataStruct(doTiming);

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	time_computeStep1 = ComputeDensityAndBuildNeighborList(doTiming);
#else
	time_computeStep1 = ComputeStep1(doTiming);
#endif

	time_ComputeStep2 = ComputeStep2(doTiming);

	time_ComputeStep3 = ComputeStep3(doTiming);

	time_integrateForces = Integrate(doTiming, progress, mSettings->GetValue("Timestep"), gridWallCollisions, terrainCollisions, fluidWorldPosition, dTerrainData);
	
	thrust::device_ptr<float> dev_ptr(mSPHBuffers->Get(BufferCFLSorted)->GetPtr<float>());
	float maxVel = thrust::reduce(dev_ptr, dev_ptr + mNumParticles, -1.0f,  thrust::maximum<float>());;

// 	float cflVelocityTerm = mCudaMaxScan->FindMax(mSPHBuffers->Get(BufferCFLSorted)->GetPtr<float>());
// 	float cflVelocityTimeStep = 0.1*hFluidParams.smoothing_length / cflVelocityTerm;
// 	float cflViscosityTimeStep = 2;//0.125*(hFluidParams.smoothing_length*hFluidParams.smoothing_length) / (6*200);
// 	float cflTimeStep = min(cflVelocityTimeStep,cflViscosityTimeStep);

	if(doTiming)
	{
		char tmp[2048];
		sprintf(tmp,"%4.4f\t%4.4f\t%4.4f\t%4.4f\t%4.4f\t%4.4f\t%4.4f\t\n", time_hashSPHData, time_radixsort, time_updatelists, time_computeStep1, time_ComputeStep2, time_ComputeStep3, time_integrateForces);
		printf(tmp);
	}

}

void SimSnowSPH::BindTextures()
{
	SnowSPHData dParticleDataSorted = GetParticleDataSorted();

#ifdef USE_TEX
	CUDA_SAFE_CALL(cudaBindTexture(0, position_tex, dParticleDataSorted.position, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velocity_tex, dParticleDataSorted.velocity, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, veleval_tex, dParticleDataSorted.veleval, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, color_tex, dParticleDataSorted.color, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, xsph_tex, dParticleDataSorted.xsph, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, sph_force_tex, dParticleDataSorted.sph_force, mNumParticles*sizeof(float_vec)));
	CUDA_SAFE_CALL(cudaBindTexture(0, pressure_tex, dParticleDataSorted.pressure, mNumParticles*sizeof(float)));
	CUDA_SAFE_CALL(cudaBindTexture(0, density_tex, dParticleDataSorted.density, mNumParticles*sizeof(float)));
	CUDA_SAFE_CALL(cudaBindTexture(0, stress_tensor_tex, dParticleDataSorted.stress_tensor, mNumParticles*sizeof(matrix3)));

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	CUDA_SAFE_CALL(cudaBindTexture(0, neighbors_tex, dNeighborList.neighbors, dNeighborList.MAX_NEIGHBORS * dNeighborList.numParticles * sizeof(uint)));
#endif

	GridData dGridData = mUniformGrid->GetGridData();
	CUDA_SAFE_CALL(cudaBindTexture(0, cell_indexes_start_tex, dGridData.cell_indexes_start, mUniformGrid->GetNumCells() * sizeof(uint)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cell_indexes_end_tex, dGridData.cell_indexes_end, mUniformGrid->GetNumCells()  * sizeof(uint)));
#endif
}

void SimSnowSPH::UnbindTextures()
{
#ifdef USE_TEX
	CUDA_SAFE_CALL(cudaUnbindTexture(position_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velocity_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(veleval_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(xsph_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(sph_force_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(color_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(pressure_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(density_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(stress_tensor_tex));

#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	CUDA_SAFE_CALL(cudaUnbindTexture(neighbors_tex));
#endif

	CUDA_SAFE_CALL(cudaUnbindTexture(cell_indexes_start_tex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cell_indexes_end_tex));
#endif
}

SnowSPHData SimSnowSPH::GetParticleData()
{
	SnowSPHData dParticleData;
	dParticleData.color = mBaseBuffers->GetPtr<float_vec>(BufferColor);
	dParticleData.position = mBaseBuffers->GetPtr<float_vec>(BufferPosition);
	dParticleData.veleval = mBaseBuffers->GetPtr<float_vec>(BufferVeleval);
	dParticleData.velocity = mBaseBuffers->GetPtr<float_vec>(BufferVelocity);
	return dParticleData;
}

SnowSPHData SimSnowSPH::GetParticleDataSorted()
{
	SnowSPHData dParticleDataSorted;
	dParticleDataSorted.color = mBaseBuffers->GetPtr<float_vec>(BufferColorSorted);
	dParticleDataSorted.position = mBaseBuffers->GetPtr<float_vec>(BufferPositionSorted);
	dParticleDataSorted.veleval = mBaseBuffers->GetPtr<float_vec>(BufferVelevalSorted);
	dParticleDataSorted.velocity = mBaseBuffers->GetPtr<float_vec>(BufferVelocitySorted);

	dParticleDataSorted.xsph = mSPHBuffers->GetPtr<float_vec>(BufferXSPHSorted);
	dParticleDataSorted.sph_force = mSPHBuffers->GetPtr<float_vec>(BufferSphForceSorted);
	dParticleDataSorted.pressure = mSPHBuffers->GetPtr<float>(BufferSphPressureSorted);
	dParticleDataSorted.density = mSPHBuffers->GetPtr<float>(BufferSphDensitySorted);
	dParticleDataSorted.stress_tensor = mSPHBuffers->GetPtr<matrix3>(BufferSphStressTensorSorted);
	return dParticleDataSorted;
}

float SimSnowSPH::BuildDataStruct(bool doTiming)
{
	GridData dGridData = mUniformGrid->GetGridData();
	SnowSPHData dParticleData = GetParticleData();
	SnowSPHData dParticleDataSorted = GetParticleDataSorted();

	uint threadsPerBlock;

	//TODO; this is not correct, need to calculate based on actual device parameters...
	if(mSimCudaHelper->IsFermi())
	{
		//Used 7 registers, 160+16 bytes smem, 140 bytes cmem[0], 4 bytes cmem[1]
		threadsPerBlock = 256;
	}
	else {
		// Used 7 registers, 192+16 bytes smem, 156 bytes cmem[0], 4 bytes cmem[1]
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

	K_Grid_UpdateSorted<SnowSPHSystem, SnowSPHData><<< numBlocks, numThreads, smemSize>>> (
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

float SimSnowSPH::ComputeDensityAndBuildNeighborList(bool doTiming)
{
	GridData dGridData = mUniformGrid->GetGridData();
	SnowSPHData dParticleData = GetParticleData();
	SnowSPHData dParticleDataSorted = GetParticleDataSorted();

	//Used 27 registers, 144+16 bytes smem, 156 bytes cmem[0], 8 bytes cmem[1], 8 bytes cmem[14]
	uint numThreads, numBlocks;
	computeGridSize(mNumParticles, 256, numBlocks, numThreads);

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

float SimSnowSPH::ComputeStep1(bool doTiming)
{
	GridData dGridData = mUniformGrid->GetGridData();
	SnowSPHData dParticleDataSorted = GetParticleDataSorted();

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
	// Used 25 registers, 144+16 bytes smem, 160 bytes cmem[0], 8 bytes cmem[1], 8 bytes cmem[14]
	threadsPerBlock = 288;
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


float SimSnowSPH::ComputeStep2(bool doTiming)
{
	GridData dGridData = mUniformGrid->GetGridData();
	SnowSPHData dParticleDataSorted = GetParticleDataSorted();

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
		//sm_20:Used 38 registers, 168 bytes cmem[0], 36 bytes cmem[2], 8 bytes cmem[14], 4 bytes cmem[16]
		threadsPerBlock = 416;
	}
	else 
	{
		// Used 38 registers, 168 bytes cmem[0], 36 bytes cmem[2], 8 bytes cmem[14], 4 bytes cmem[16]
		threadsPerBlock = 448;
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

	K_SumStep2<<<numBlocks, numThreads>>>(
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

float SimSnowSPH::ComputeStep3(bool doTiming)
{
	GridData dGridData = mUniformGrid->GetGridData();
	SnowSPHData dParticleDataSorted = GetParticleDataSorted();

	uint threadsPerBlock;
#ifdef SPHSIMLIB_USE_NEIGHBORLIST
#ifdef SPHSIMLIB_USE_NEIGHBORLIST_PRECALC_R
	//sm_13: Used 25 registers, 144+16 bytes smem, 156 bytes cmem[0], 4 bytes cmem[1], 8 bytes cmem[14]
	threadsPerBlock = 320;
#else
	//sm_13: Used 27 registers, 144+16 bytes smem, 156 bytes cmem[0], 4 bytes cmem[1], 8 bytes cmem[14]
	threadsPerBlock = 64;
#endif
#else
	//TODO; this is not correct, need to calculate based on actual device parameters...
	if(mSimCudaHelper->IsFermi())
	{
		//sm_20: Used 32 registers, 176+0 bytes lmem, 168 bytes cmem[0], 24 bytes cmem[2], 8 bytes cmem[14]
		threadsPerBlock = 256;
	}
	else 
	{
		//sm_13: Used 55 registers, 144+16 bytes smem, 160 bytes cmem[0], 4 bytes cmem[1], 8 bytes cmem[14]
		threadsPerBlock = 64;
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

	K_SumStep3<<<numBlocks, numThreads>>>(
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

float SimSnowSPH::Integrate(bool doTiming, bool progress, float deltaTime, bool gridWallCollisions, bool terrainCollisions, float3 fluidWorldPosition, TerrainData dTerrainData)
{
	GridData	dGridData = mUniformGrid->GetGridData();
	SnowSPHData dParticleData = GetParticleData();
	SnowSPHData dParticleDataSorted = GetParticleDataSorted();

	if(doTiming)
	{
		mGPUTimer->start();
	}

	int threadsPerBlock;

	//TODO; this is not correct, need to calculate based on actual device parameters...
	if(mSimCudaHelper->IsFermi())
	{
		//sm_20: Used 29 registers, 280 bytes cmem[0], 36 bytes cmem[2], 8 bytes cmem[14], 4 bytes cmem[16]
		threadsPerBlock = 256;
	}
	else 
	{
		//sm_13: Used 27 registers, 256+16 bytes smem, 144 bytes cmem[0], 16 bytes cmem[1]
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
		//,mSPHBuffers->Get(BufferCFLSorted)->GetPtr<float>()
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
