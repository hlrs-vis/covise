#ifndef __SimSimpleSPH_cuh__
#define __SimSimpleSPH_cuh__

#include "K_Common.cuh"

#include "SimCudaAllocator.h"
#include "SimBase.cuh"
#include "UniformGrid.cuh"

#include "K_Coloring.cuh"
#include "K_SPH_Kernels.inl"
#include "UniformGrid.cuh"
#include "SimulationSystem.h"
#include "timer.h"
#include "K_SPH_Common.cuh"

typedef unsigned int uint;



namespace SimLib { namespace Sim { namespace SimpleSPH { 


	enum SimpleSPHBuffers
	{
		BufferSphForce,
		BufferSphPressure,
		BufferSphDensity,
		BufferSphForceSorted,
		BufferSphPressureSorted,
		BufferSphDensitySorted
	};

	struct SimpleSPHPrecalcParams
	{
		// smoothing length^2 
		float			smoothing_length_pow2;

		// precalculated terms for smoothing kernels
		float			kernel_poly6_coeff;
		float			kernel_spiky_grad_coeff;
		float			kernel_viscosity_lap_coeff;	
		float			kernel_pressure_precalc;
		float			kernel_viscosity_precalc;
	};

	struct SimpleSPHFluidParams 
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

		// internal stiffness in fluid
		float			gas_stiffness;		

		// external stiffness (against boundaries)
		float			boundary_stiffness; 

		// external dampening (against boundaries)
		float			boundary_dampening;

		// the distance a particle will be "pushed" away from boundaries
		float			boundary_distance;

		// velocity limit of particle (artificial dampening of system)
		float			velocity_limit;

		float			friction_static_limit;
		float			friction_kinetic;

	};

	struct SimpleSPHData : ParticleData
	{
		// sum of sph forces
		float_vec* sph_force;

		// sph pressure
		float* pressure;

		// sph density
		float* density;
	};


class SimSimpleSPH : public SimBase
{
public:
	SimSimpleSPH(SimLib::SimCudaAllocator* SimCudaAllocator, SimLib::SimCudaHelper* simCudaHelper);
	~SimSimpleSPH();

	void Clear();

	void Alloc(uint numParticles);
	void Simulate(bool doTiming, bool progress, bool gridWallCollisions, bool terrainCollisions, float3 fluidWorldPosition, TerrainData dTerrainData);

	SimpleSPHFluidParams& GetFluidParams();

	float GetParticleSize();
	float GetParticleSpacing();

	SimpleSPHData GetParticleData();
	SimpleSPHData GetParticleDataSorted();

private:
	bool mAlloced;
	bool mParams;

	ocu::GPUTimer *mGPUTimer;

protected:
	void SettingChanged(std::string settingName);
	void Free();
	void UpdateParams();

	float BuildDataStruct(bool doTiming);
	float ComputeDensityAndBuildNeighborList(bool doTiming);
	float ComputeStep1(bool doTiming);
	float ComputeStep2(bool doTiming);
	float Integrate(bool doTiming, bool progress, float deltaTime, bool gridWallCollisions, bool terrainCollisions, float3 fluidWorldPosition, TerrainData dTerrainData);

	void BindTextures();
	void UnbindTextures();

	SPHSymmetrization mSymmetrizationType;

	// params
	SimpleSPHFluidParams hFluidParams;
	SimpleSPHPrecalcParams hPrecalcParams;

	SimLib::BufferManager<SimpleSPHBuffers>*	mSPHBuffers;
};

}}} // namespace SimLib { namespace Sim { namespace SimpleSPH { 

#endif