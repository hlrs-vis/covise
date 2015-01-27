#ifndef __K_SnowSPH_Force_cu__
#define __K_SnowSPH_Force_cu__

#include "K_UniformGrid_Utils.inl"
#include "K_SPH_Kernels.inl"
#include "K_SPH_Common.inl"
#include "K_Common.cuh"

class Step2
{
public:
	struct Data
	{
		float density_i;
		float3 veleval_i;
		matrix3 sum_velocity_tensor;

		SnowSPHData dParticleDataSorted;
	};

	class Calc
	{
	public:
		// this is called before the loop over each neighbor particle
		static __device__ void PreCalc(Data &data, uint index_i)
		{
			// read particle data from sorted arrays
			data.density_i	= FETCH(data.dParticleDataSorted, density, index_i);
			data.veleval_i	= FETCH_FLOAT3(data.dParticleDataSorted, veleval, index_i);
			data.sum_velocity_tensor = make_matrix3(0,0,0,0,0,0,0,0,0);
		}

		static __device__ void ForNeighbor(Data &data, uint const &index_i, uint const &index_j, float3 const &r, float const& rlen, float const &rlen_sq)
		{
			float density_j	= FETCH(data.dParticleDataSorted, density, index_j);
			float3 veleval_j = FETCH_FLOAT3(data.dParticleDataSorted, veleval, index_j);

			float3 gradW = SPH_Kernels::Wcubic::Gradient(cFluidParams.smoothing_length, cPrecalcParams.smoothing_length_pow2,  cPrecalcParams.smoothing_length_pow3, cPrecalcParams.smoothing_length_pow4,   r, rlen, rlen_sq);

			// calculate the velocity tensor sum
			data.sum_velocity_tensor += outer(
				(veleval_j  - data.veleval_i)/(density_j)
				, gradW
				);
		}


		// this is called after the loop over each particle in a cell
		static __device__ void PostCalc(Data &data, uint index_i)
		{
			//data.sum_velocity_tensor *= SPH_Kernels::Wviscosity::Gradient_Constant(cFluidParams.smoothing_length);

			// velocity tensor derivative (DELv)
			matrix3 velocity_tensor_i = cFluidParams.particle_mass * data.sum_velocity_tensor;

			// rate-of-deformation/rate-of-strain tensor (E on wiki(NSE), D in viscoplastic paper)
			matrix3 deformation_tensor_i = 0.5*(velocity_tensor_i + transpose(velocity_tensor_i));
			//matrix3 deformation_tensor_i = (velocity_tensor_i + transpose(velocity_tensor_i));

			// from "Particle-based viscoplastic fluid/solid simulation"
			float t = trace(deformation_tensor_i);
			float deformation_amount = sqrtf(t*t);

			// stress tensor
			matrix3 stress_tensor;// = make_matrix3(0,0,0,0,0,0,0,0,0);

			//viscoplastic fluid (exp-power model w/jump number for melting stuff.. e.g. lava)
 			//float n = 0.5f;
 			//float J = 10;
 			//stress_tensor = (1-__expf(-(J+1)*deformation_amount))*(pow(deformation_amount, n-1.0f)+1/deformation_amount)*deformation_amount;

			// newtonian fluid
			// 3-step says: ( t = 2*µ*D )
			//stress_tensor = 1*deformation_amount*deformation_tensor_i;

			// non-newtonian POWER-LAW fluid
// 			float n = 3.2;
// 			float K = 1.74;
// 			float viscosity = K*pow(deformation_amount,n-1.0f);
// 			viscosity = clamp(viscosity, 1.0f,300.0f);
// 	  		stress_tensor = viscosity*deformation_tensor_i;

			// non-newtonian BINGHAM fluid
// 			float K = 10;
//  			float yield_stress = 1.5;
//  			stress_tensor = yield_stress + K * deformation_tensor_i;
// 			float s = trace(stress_tensor);
// 			float stress_amount = sqrtf(s*s);
//  			if(stress_amount <= yield_stress)
//  			{
// 				stress_tensor = 500*deformation_amount*deformation_tensor_i;
//  			}


			// non-newtonian HERSCHEL-BULKLEY fluid
//  			float K = 1;
//  			float yield_stress = 1.5;
//  			float n = 1.74;
//  			stress_tensor = (K*pow(deformation_amount,n-1.0f) +yield_stress/deformation_amount)*deformation_tensor_i;
//  			float stress_amount = trace(stress_tensor)/3.0f;
//  			if(stress_amount < yield_stress) {
// 				stress_tensor = 500*deformation_amount*deformation_tensor_i;
//  			}

			// non-newtonian cross fluid
  			float K = 2.1f;
 			float visco_inf= 1.07;
 			float visco_zero = 300;
 			float n = 1.0f;
 			float viscosity = visco_inf+(visco_zero-visco_inf)/(1+K*pow(deformation_amount, n));
			viscosity = clamp(viscosity, 1.0f,300.0f);
 			stress_tensor = viscosity*deformation_tensor_i;

			// store stress tensor
			data.dParticleDataSorted.stress_tensor[index_i] = stress_tensor;

			//data.dParticleDataSorted.color[index_i] = make_vec(deformation_amount,0.3,0.3);
		}
	};
};

__global__ void K_SumStep2(uint			numParticles,
						NeighborList	dNeighborList,
						SnowSPHData	dParticleDataSorted,
						GridData		dGridData
						)
{
	// particle index
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	Step2::Data data;

	data.dParticleDataSorted = dParticleDataSorted;

	float3 position_i = FETCH_FLOAT3(data.dParticleDataSorted, position, index);

	// Do calculations on particles in neighboring cells
#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	UniformGridUtils::IterateParticlesInNearbyCells<SPHNeighborCalc<Step2::Calc, Step2::Data>, Step2::Data>(data, index, position_i, dNeighborList);
#else
	UniformGridUtils::IterateParticlesInNearbyCells
		<
		SPHNeighborCalc
			<Step2::Calc, Step2::Data>
			,
			Step2::Data
		>
	(data, index, position_i, dGridData);
#endif
}


#endif