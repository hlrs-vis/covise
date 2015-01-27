#ifndef __K_SnowSPH_Density_cu__
#define __K_SnowSPH_Density_cu__

#include "K_UniformGrid_Utils.inl"
#include "K_SPH_Kernels.inl"
#include "K_SPH_Common.inl"

class Step1
{
public:

	struct Data
	{
		float sum_density;

		SnowSPHData dParticleDataSorted;
	};

	class Calc
	{
	public:

		static __device__ void PreCalc(Data &data, uint const &index_i)
		{
			// read particle data from sorted arrays
			data.sum_density = 0;
		}

		static __device__ void ForNeighbor(Data &data, uint const &index_i, uint const &index_j, float3 const &r, float const& rlen, float const &rlen_sq)
		{
			// the density sum using Wpoly6 kernel
			data.sum_density += SPH_Kernels::Wpoly6::Kernel_Variable(cPrecalcParams.smoothing_length_pow2, r, rlen_sq);	

			//data.sum_density += SPH_Kernels::Wcubic::Kernel(cPrecalcParams.smoothing_length_pow2, r, rlen_sq);	
		}

		static __device__ void PostCalc(Data &data, uint index_i)
		{
			data.sum_density *= cFluidParams.particle_mass * cPrecalcParams.kernel_poly6_coeff;
			//data.sum_density *= cFluidParams.particle_mass;

			// Compute the density field at the current particle,
			// Calculate the W smoothing function for this particle, mass and the poly6_grad_coeff has been moved outside the sum because they are constant.
			//float density = max(1.0, data.sum_density);
			data.dParticleDataSorted.density[index_i]= data.sum_density;
				
			// ideal gas equation of state (by Desbrun and Cani in "Smoothed particles: A new paradigm for animating highly deformable bodies")
			data.dParticleDataSorted.pressure[index_i] = cFluidParams.rest_pressure + cFluidParams.gas_stiffness * (data.sum_density - cFluidParams.rest_density);
		}

	};
};


__global__ void K_SumStep1(uint			numParticles,
					   NeighborList		dNeighborList, 
					   SnowSPHData		dParticleDataSorted,
					   GridData const	dGridData
					   )								
{
	// particle index	
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;		
	if (index >= numParticles) return;

	Step1::Data data;
	data.dParticleDataSorted = dParticleDataSorted;

	float3 position_i = make_float3(FETCH(dParticleDataSorted, position, index));

	// Do calculations on particles in neighboring cells
#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	UniformGridUtils::IterateParticlesInNearbyCells<SPHNeighborCalc<Step1::Calc, Step1::Data>, Step1::Data>(data, index, position_i, dNeighborList);	
#else
	UniformGridUtils::IterateParticlesInNearbyCells<SPHNeighborCalc<Step1::Calc, Step1::Data>, Step1::Data>(data, index, position_i, dGridData);
#endif

}

#endif