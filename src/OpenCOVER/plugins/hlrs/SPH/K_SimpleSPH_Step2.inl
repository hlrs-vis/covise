#ifndef __K_SimpleSPH_Step2_cu__
#define __K_SimpleSPH_Step2_cu__

#include "K_UniformGrid_Utils.inl"
#include "K_SPH_Kernels.inl"
#include "K_SPH_Common.inl"


class Step2
{
public:

	struct Data
	{
		float3 veleval_i;
		float density_i;
		float pressure_i;

		float3 veleval_j;
		float density_j;
		float pressure_j;

		float3 f_viscosity;
		float3 f_pressure;

		SimpleSPHData dParticleDataSorted;
	};

	template <SPHSymmetrization symmetrizationType>
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

			data.f_pressure		= make_float3(0,0,0);
			data.f_viscosity	= make_float3(0,0,0);
		}

		static __device__ void ForNeighbor(Data &data, uint const &index_i, uint const &index_j, float3 const &r, float const& rlen, float const &rlen_sq)
		{
			data.veleval_j	= FETCH_FLOAT3(data.dParticleDataSorted, veleval, index_j);
			data.density_j	= FETCH(data.dParticleDataSorted, density,  index_j);
			data.pressure_j	= FETCH(data.dParticleDataSorted, pressure, index_j);


			// pressure  force calc
			switch (symmetrizationType)
			{
				//mueller symmetrization of density
			case SPH_PRESSURE_MUELLER:
				{
					// in the mueller paper, density_i is placed outside the force defs..., but we calc it here.. easier(atm)
					// from paper:  f_pressure = -(1/rho_i)* SUM(m_j * ((p_i + p_j) / (2rho_j)) DELWpress
					// we move the mass the 1/2 and the Wpress constants to precalc.
					data.f_pressure  += ( (data.pressure_i + data.pressure_j) / (data.density_j * data.density_i) ) * SPH_Kernels::Wspiky::Gradient_Variable(cFluidParams.smoothing_length, r, rlen);
				}
				break;
				//from "Particle-based viscoplastic fluid/solid simulation", also see "SPH survival kit"
			case SPH_PRESSURE_VISCOPLASTIC:
				{
					data.f_pressure  += ( (data.pressure_i/(data.density_i*data.density_i)) + (data.pressure_j/(data.density_j*data.density_j)) ) * SPH_Kernels::Wspiky::Gradient_Variable(cFluidParams.smoothing_length, r, rlen);	
					break;
				}   
			}

			// viscosity from mueller paper : f_viscosity = (µ/rho_i)SUM(m_j * (v_j-v_i)/(rho_j)DEL^2Wvis
			// we move the mass and the Wvis constants to precalc
			data.f_viscosity += ( (data.veleval_j  - data. veleval_i ) / (data.density_j * data.density_i) ) * SPH_Kernels::Wviscosity::Laplace_Variable(cFluidParams.smoothing_length, r, rlen);
		}

		// this is called after the loop over each particle in a cell
		static __device__ void PostCalc(Data &data, uint index_i)
		{
			float3 sum_sph_force = (cPrecalcParams.kernel_pressure_precalc * data.f_pressure + cPrecalcParams.kernel_viscosity_precalc * data.f_viscosity );

			// Calculate the force, the particle_mass is added here because it is constant and thus there is no need for it to be inside the sum loop.
			data.dParticleDataSorted.sph_force[index_i] = make_vec(sum_sph_force * cFluidParams.particle_mass);
		}
	};

};

template <SPHSymmetrization symmetrization> 
__global__ void K_SumStep2(uint			numParticles,
						   SimpleSPHData	dParticleDataSorted,
#ifdef SPHSIMLIB_USE_NEIGHBORLIST
						   NeighborList	dNeighborList, 
#else
							 GridData		dGridData
#endif
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
	UniformGridUtils::IterateParticlesInNearbyCells<SPHNeighborCalc<Step2::Calc<symmetrization>, Step2::Data>, Step2::Data>(data, index, position_i, dNeighborList);
#else
	UniformGridUtils::IterateParticlesInNearbyCells<SPHNeighborCalc<Step2::Calc<symmetrization>, Step2::Data>, Step2::Data>(data, index, position_i, dGridData);
#endif
}

#endif