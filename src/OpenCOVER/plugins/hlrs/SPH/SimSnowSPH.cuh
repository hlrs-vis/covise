#ifndef __SimSnowSPH_cuh__
#define __SimSnowSPH_cuh__

#include "K_Common.cuh"

#include "SimCudaAllocator.h"
#include "SimBase.cuh"
#include "UniformGrid.cuh"

#include "K_Coloring.cuh"
#include "K_SPH_Kernels.inl"
#include "UniformGrid.cuh"
#include "SimulationSystem.h"
#include "K_SPH_Common.cuh"

typedef unsigned int uint;

namespace SimLib { namespace Sim { namespace SnowSPH { 


	enum SnowSPHBuffers
	{
		BufferXSPHSorted
		, BufferSphForceSorted
		, BufferSphPressureSorted
		, BufferSphDensitySorted
		, BufferSphStressTensorSorted
		, BufferCFLSorted
	};


	struct SnowSPHPrecalcParams
	{
		// smoothing length^2 
		float			smoothing_length_pow2;
		float			smoothing_length_pow3;
		float			smoothing_length_pow4;
		float			smoothing_length_pow5;

		// precalculated terms for smoothing kernels
		float			kernel_poly6_coeff;
		float			kernel_spiky_grad_coeff;
		float			kernel_viscosity_lap_coeff;	
		float			kernel_pressure_precalc;
		float			kernel_viscosity_precalc;
	};

	struct SnowSPHFluidParams 
	{
		// the smoothing length of the kernels
		float			smoothing_length; 

		// the "ideal" rest state distance between each particle
		float			particle_rest_distance;

		// the scale of the simulation (ie. 0.1 of world scale)
		float			scale_to_simulation;

		// the mass of each particle
		float			particle_mass;

		// pressure calculation parameters
		float    		rest_density;
		float			rest_pressure;

		// viscosity of fluid
		float			viscosity;

		// internal stiffness in fluid ( used by EOS )
		float			gas_stiffness;		

		// external stiffness (against boundaries)
		float			boundary_stiffness; 

		// external dampening (against boundaries)
		float			boundary_dampening;

		// the distance a particle will be "pushed" away from boundaries
		float			boundary_distance;

		float			friction_static_limit;
		float			friction_kinetic;

		// velocity limit of particle (artificial dampening of system)
		float			velocity_limit;

		//
		float			xsph_factor;

	};

	struct SnowSPHData : ParticleData
	{
		float_vec* xsph;

		// sum of sph forces
		float_vec* sph_force;

		// sph pressure
		float* pressure;

		// sph density
		float* density;

		// NSE/viscosity stress tensor
		matrix3* stress_tensor;
	};

class SimSnowSPH : public SimBase
{
public:
	SimSnowSPH(SimLib::SimCudaAllocator* simCudaAllocator, SimLib::SimCudaHelper* simCudaHelper);
	~SimSnowSPH();

	void Clear();

	void Alloc(uint numParticles);
	void Simulate(bool doTiming, bool progress, bool gridWallCollisions, bool terrainCollisions, float3 fluidWorldPosition, TerrainData dTerrainData);

	SnowSPHFluidParams& GetFluidParams();

	float GetParticleSize();
	float GetParticleSpacing();

	SnowSPHData GetParticleData();
	SnowSPHData GetParticleDataSorted();

private:
	bool mAlloced;
	bool mParams;

protected:
	void SettingChanged(std::string settingName);
	void Free();
	void UpdateParams();

	float BuildDataStruct(bool doTiming);
	float ComputeDensityAndBuildNeighborList(bool doTiming);
	float ComputeStep1(bool doTiming);
	float ComputeStep2(bool doTiming);
	float ComputeStep3(bool doTiming);
	float Integrate(bool doTiming, bool progress, float deltaTime, bool gridWallCollisions, bool terrainCollisions, float3 fluidWorldPosition, TerrainData dTerrainData);

	void BindTextures();
	void UnbindTextures();


	// params
	SnowSPHFluidParams hFluidParams;
	SnowSPHPrecalcParams hPrecalcParams;

	SimLib::BufferManager<SnowSPHBuffers>*	mSPHBuffers;
};

}}} // namespace SimLib { namespace Sim { namespace SnowPH { 
#endif