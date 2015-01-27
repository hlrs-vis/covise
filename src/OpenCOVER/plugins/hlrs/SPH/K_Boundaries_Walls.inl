#ifndef __K_Boundaries_Walls_cu__
#define __K_Boundaries_Walls_cu__

#include "K_Boundaries_Common.inl"

#define EPSILON			0.00001f			//for collision detection

__device__ float3 calculateWallsNoPenetrationForce(
	float3 const& pos, 
	float3 const& vel,
	float3 const& grid_min, 
	float3 const& grid_max, 
	float const& boundary_distance,
	float const& boundary_stiffness,
	float const& boundary_dampening,
	float const& scale_to_simulation)
{
	float3 repulsion_force = make_float3(0,0,0);
	float diff;

	// simple limit for "wall" in Y direction (min of simulated volume)
	diff = boundary_distance - ((pos.y - grid_min.y ) * scale_to_simulation);			
	if (diff > EPSILON) {
		float3 normal = make_float3(0,1,0);
		repulsion_force  += calculateRepulsionForce(vel, normal, diff, boundary_dampening, boundary_stiffness);
	}

	// simple limit for "wall" in Y direction (max of simulated volume)
	diff = boundary_distance - ((grid_max.y - pos.y ) * scale_to_simulation);
	if (diff > EPSILON) {
		float3 normal = make_float3(0,-1,0);
		repulsion_force  += calculateRepulsionForce(vel, normal, diff, boundary_dampening, boundary_stiffness);
	}

	// simple limit for "wall" in Z direction (min of simulated volume)
	diff = boundary_distance - ((pos.z - grid_min.z ) * scale_to_simulation);
	if (diff > EPSILON ) {			
		float3 normal = make_float3(0,0,1);
		repulsion_force  += calculateRepulsionForce(vel, normal, diff, boundary_dampening, boundary_stiffness);
	}		

	// simple limit for "wall" in Z direction (max of simulated volume)
	diff = boundary_distance - ((grid_max.z - pos.z ) * scale_to_simulation);
	if (diff > EPSILON) {
		float3 normal = make_float3(0,0,-1);
		float adj =  boundary_stiffness * diff - boundary_dampening * dot(normal, vel);
		repulsion_force  += adj * normal;
	}

	// simple limit for "wall" in X direction (min of simulated volume)
	diff = boundary_distance - ((pos.x - grid_min.x ) * scale_to_simulation);
	if (diff > EPSILON ) {			
		float3 normal = make_float3(1,0,0);
		repulsion_force  += calculateRepulsionForce(vel, normal, diff, boundary_dampening, boundary_stiffness);
	}			

	// simple limit for "wall" in X direction (max of simulated volume)
	diff = boundary_distance - ((grid_max.x - pos.x ) * scale_to_simulation);
	if (diff > EPSILON) {
		float3 normal = make_float3(-1,0,0);
		repulsion_force  += calculateRepulsionForce(vel, normal, diff, boundary_dampening, boundary_stiffness);
	}	

	return repulsion_force;
}


__device__ float3 calculateWallsNoSlipForce(
	float3 const& pos, 
	float3 const& vel,
	float3 const& force,
	float3 const& grid_min, 
	float3 const& grid_max, 
	float const& boundary_distance,
	float const& friction_kinetic,
	float const& friction_static_limit,
	float const& scale_to_simulation)
{
	float3 friction_force = make_float3(0,0,0);
	float diff;

	// simple limit for "wall" in Y direction (min of simulated volume)
	diff = boundary_distance - ((pos.y - grid_min.y ) * scale_to_simulation);			
	if (diff > EPSILON) {
		float3 normal = make_float3(0,1,0);
		friction_force += calculateFrictionForce(vel, force, normal, friction_kinetic, friction_static_limit);
	}

	// simple limit for "wall" in Y direction (max of simulated volume)
	diff = boundary_distance - ((grid_max.y - pos.y ) * scale_to_simulation);
	if (diff > EPSILON) {
		float3 normal = make_float3(0,-1,0);
		friction_force += calculateFrictionForce(vel, force, normal, friction_kinetic, friction_static_limit);
	}

	// simple limit for "wall" in Z direction (min of simulated volume)
	diff = boundary_distance - ((pos.z - grid_min.z ) * scale_to_simulation);
	if (diff > EPSILON ) {			
		float3 normal = make_float3(0,0,1);
		friction_force += calculateFrictionForce(vel, force, normal, friction_kinetic, friction_static_limit);
	}		

	// simple limit for "wall" in Z direction (max of simulated volume)
	diff = boundary_distance - ((grid_max.z - pos.z ) * scale_to_simulation);
	if (diff > EPSILON) {
		float3 normal = make_float3(0,0,-1);
		friction_force += calculateFrictionForce(vel, force, normal, friction_kinetic, friction_static_limit);
	}

	// simple limit for "wall" in X direction (min of simulated volume)
	diff = boundary_distance - ((pos.x - grid_min.x ) * scale_to_simulation);
	if (diff > EPSILON ) {			
		float3 normal = make_float3(1,0,0);
		friction_force += calculateFrictionForce(vel, force, normal, friction_kinetic, friction_static_limit);
	}			

	// simple limit for "wall" in X direction (max of simulated volume)
	diff = boundary_distance - ((grid_max.x - pos.x ) * scale_to_simulation);
	if (diff > EPSILON) {
		float3 normal = make_float3(-1,0,0);
		friction_force += calculateFrictionForce(vel, force, normal, friction_kinetic, friction_static_limit);
	}	

	return friction_force;
}

#endif