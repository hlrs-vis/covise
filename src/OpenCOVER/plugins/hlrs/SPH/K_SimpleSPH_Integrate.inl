#ifndef __K_SimpleSPH_Integrate_cu__
#define __K_SimpleSPH_Integrate_cu__

#include "K_Coloring.inl"
#include "K_Boundaries_Terrain.inl"
#include "K_Boundaries_Walls.inl"

template<SPHColoringSource coloringSource, ColoringGradient coloringGradient>
__global__ void K_Integrate(int				numParticles,
								bool			gridWallCollisions,
								bool			terrainCollisions,
								float			delta_time,
								bool			progress,	
								GridData		dGridData,
								SimpleSPHData	dParticleData, 
								SimpleSPHData	dParticleDataSorted, 
								float3			fluidWorldPosition,
								TerrainData		dTerrainData
								) 
{
	int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	float3 pos			= make_float3(FETCH_NOTEX(dParticleDataSorted, position, index));
	float3 vel			= make_float3(FETCH_NOTEX(dParticleDataSorted, velocity, index));
	float3 vel_eval		= make_float3(FETCH_NOTEX(dParticleDataSorted, veleval, index));

	float3 sph_force	= make_float3(FETCH_NOTEX(dParticleDataSorted, sph_force, index));
	float  sph_pressure	= FETCH_NOTEX(dParticleDataSorted, pressure, index);
	//float  sph_density	= FETCH_NOTEX(dParticleDataSorted, density, index);
	
// 	if(pos.x < cGridParams.grid_max.x/3 && pos.z < cGridParams.grid_max.z/3)
// 	{
// 		// negate gravity	
// 		//external_force.y += 9.8f;	
// 
// 		external_force.x += 3.f;	
// 		external_force.y += 12.f;	
// 		external_force.z += 2.f;	
// 	}

	float3 external_force = make_float3(0,0,0);
	// add gravity	
	external_force.y -= 9.8f;	


	// add no-penetration force due to terrain
	if(terrainCollisions)
		external_force += calculateTerrainNoPenetrationForce(
			pos, vel_eval, 
			fluidWorldPosition, dTerrainData,
			cFluidParams.boundary_distance,
			cFluidParams.boundary_stiffness,
			cFluidParams.boundary_dampening,
			cFluidParams.scale_to_simulation);

// 	// add no-slip force due to terrain..
	if(terrainCollisions)
		external_force += calculateTerrainFrictionForce(
			pos, vel_eval, sph_force+external_force,
			fluidWorldPosition, dTerrainData,
			cFluidParams.boundary_distance,
			cFluidParams.friction_kinetic/delta_time,
			cFluidParams.friction_static_limit,
			cFluidParams.scale_to_simulation);


	// add no-penetration force due to "walls"
	if(gridWallCollisions)
		external_force += calculateWallsNoPenetrationForce(
			pos, vel_eval,
			cGridParams.grid_min, 
			cGridParams.grid_max,
			cFluidParams.boundary_distance,
			cFluidParams.boundary_stiffness,
			cFluidParams.boundary_dampening,
			cFluidParams.scale_to_simulation);

	// add no-slip force due to "walls"
	if(gridWallCollisions)
		external_force += calculateWallsNoSlipForce(
			pos, vel_eval, sph_force + external_force,
			cGridParams.grid_min, 
			cGridParams.grid_max,
			cFluidParams.boundary_distance,
			cFluidParams.friction_kinetic/delta_time,
			cFluidParams.friction_static_limit,
			cFluidParams.scale_to_simulation);

	float3 force = sph_force + external_force;

	// limit velocity
	float speed = length(force);
	if (speed > cFluidParams.velocity_limit ) {
		force *= cFluidParams.velocity_limit / speed;
	}

	// Leapfrog integration		
	// v(t+1/2) = v(t-1/2) + a(t) dt	
	float3 vnext = vel + force * delta_time;
	// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5
	vel_eval = (vel + vnext) * 0.5;
	vel = vnext;

	// update position of particle
	pos += vnext * (delta_time / cFluidParams.scale_to_simulation);

	if(progress)
	{
		uint originalIndex = dGridData.sort_indexes[index];

		// writeback to unsorted buffer		
		dParticleData.position[originalIndex]	= make_vec(pos);
		dParticleData.velocity[originalIndex]	= make_vec(vel);
		dParticleData.veleval[originalIndex]	= make_vec(vel_eval);

		float3 color = CalculateColor(coloringGradient, coloringSource, vnext, sph_pressure, sph_force);
		dParticleData.color[originalIndex]	= make_float4(color, 1);
	}
}

#endif