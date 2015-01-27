#ifndef __K_SnowSPH_Step3_cu__
#define __K_SnowSPH_Step3_cu__

#include "K_UniformGrid_Utils.inl"
#include "K_SPH_Kernels.inl"
#include "K_SPH_Common.inl"

class Step3
{
public:

	struct Data
	{
		float3 veleval_i;
		float density_i;
		float pressure_i;
		matrix3 stress_tensor_i;

		float3 veleval_j;
		float density_j;
		float pressure_j;
		matrix3 stress_tensor_j;


		float3 f_viscosity;
		float3 f_pressure;
		float3 f_stress;
		float3 f_xsph;
		

		SnowSPHData dParticleDataSorted;
	};

	class Calc
	{	
	public:

		// this is called before the loop over each neighbor particle
		static __device__ void PreCalc(Data &data, uint index_i)
		{
			// read particle data from sorted arrays
			data.veleval_i	= FETCH_FLOAT3(data.dParticleDataSorted, veleval, index_i);
			data.density_i	= FETCH(data.dParticleDataSorted, density, index_i);
			data.pressure_i	= FETCH(data.dParticleDataSorted, pressure, index_i);
			data.stress_tensor_i	= FETCH_MATRIX3(data.dParticleDataSorted, stress_tensor, index_i);

			data.f_pressure		= make_float3(0,0,0);
			data.f_viscosity	= make_float3(0,0,0);
			data.f_stress		= make_float3(0,0,0);
			data.f_xsph			= make_float3(0,0,0);
		}

		static __device__ void ForNeighbor(Data &data, uint const &index_i, uint const &index_j, float3 const &r, float const& rlen, float const &rlen_sq)
		{
			data.veleval_j	= FETCH_FLOAT3(data.dParticleDataSorted, veleval, index_j);
			data.density_j	= FETCH(data.dParticleDataSorted, density,  index_j);
			data.pressure_j	= FETCH(data.dParticleDataSorted, pressure,  index_j);
			data.stress_tensor_j	= FETCH_MATRIX3(data.dParticleDataSorted, stress_tensor, index_j);

			// XSPH velocity correction, Monaghan JCP 2000
			data.f_xsph += ( (data.veleval_j - data.veleval_i) / (data.density_i+data.density_j) ) *  SPH_Kernels::Wpoly6::Kernel_Variable(cPrecalcParams.smoothing_length_pow2, r, rlen_sq);

			//from "Particle-based viscoplastic fluid/solid simulation", also see "SPH survival kit"
			data.f_pressure  += ( (data.pressure_i/(data.density_i*data.density_i)) + (data.pressure_j/(data.density_j*data.density_j)) ) * SPH_Kernels::Wspiky::Gradient_Variable(cFluidParams.smoothing_length, r, rlen);	
				
			// viscosity from mueller paper : f_viscosity = (µ/rho_i)SUM(m_j * (v_j-v_i)/(rho_j)DEL^2Wvis
			// we move the mass and the Wvis constants to precalc
			data.f_viscosity += ( (data.veleval_j  - data. veleval_i ) / (data.density_j * data.density_i) ) * SPH_Kernels::Wviscosity::Laplace_Variable(cFluidParams.smoothing_length, r, rlen);

			// stress force calculation
			data.f_stress += dot(
				(data.stress_tensor_i+data.stress_tensor_j)/(data.density_j)
				, SPH_Kernels::Wcubic::Gradient(cFluidParams.smoothing_length, cPrecalcParams.smoothing_length_pow2,  cPrecalcParams.smoothing_length_pow3, cPrecalcParams.smoothing_length_pow4,   r, rlen, rlen_sq)
				);
		}		

		// this is called after the loop over each particle in a cell
		static __device__ void PostCalc(Data &data, uint index_i)
		{
			//data.f_stress *= SPH_Kernels::Wviscosity::Gradient_Constant(cFluidParams.smoothing_length);
			data.f_stress		*= (cFluidParams.particle_mass/data.density_i);

			// Calculate the forces, the particle_mass/constants are added here because there is no need for it to be inside the sum loop.
			data.f_pressure		*= cFluidParams.particle_mass * cPrecalcParams.kernel_pressure_precalc;
			data.f_viscosity	*= cFluidParams.particle_mass * cPrecalcParams.kernel_viscosity_precalc;
			data.f_xsph			*= 2 * cFluidParams.particle_mass;

			//data.dParticleDataSorted.color[index_i] = make_vec(data.stress_tensor_i.r1.x, data.stress_tensor_i.r2.x, data.stress_tensor_i.r3.x);
			//data.dParticleDataSorted.color[index_i] = make_vec(data.f_stress.x, data.f_stress.y, data.f_stress.z);
			//data.dParticleDataSorted.color[index_i] = make_vec(1,1,1);

			// store xsph val
			data.dParticleDataSorted.xsph[index_i] = make_vec(data.f_xsph);

			float3 sph_force =  (
				data.f_pressure 
				+ data.f_stress				
				+ data.f_viscosity
				);

			data.dParticleDataSorted.sph_force[index_i] = make_vec(sph_force);
		}
	};
};



__global__ void K_SumStep3(uint			numParticles,
						NeighborList	dNeighborList, 
						SnowSPHData	dParticleDataSorted,
						GridData		dGridData
						)								
{
	// particle index	
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;		
	if (index >= numParticles) return;

	Step3::Data data;

	data.dParticleDataSorted = dParticleDataSorted;

	float3 position_i = FETCH_FLOAT3(data.dParticleDataSorted, position, index);

	// Do calculations on particles in neighboring cells
#ifdef SPHSIMLIB_USE_NEIGHBORLIST
	UniformGridUtils::IterateParticlesInNearbyCells<SPHNeighborCalc<Step3::Calc, Step3::Data>, Step3::Data>(data, index, position_i, dNeighborList);
#else
	UniformGridUtils::IterateParticlesInNearbyCells<SPHNeighborCalc<Step3::Calc, Step3::Data>, Step3::Data>(data, index, position_i, dGridData);
#endif
}

#endif