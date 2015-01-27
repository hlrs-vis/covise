#ifndef __K_Boundaries_Common_cu__
#define __K_Boundaries_Common_cu__


#define EPSILON			0.00001f			//for collision detection


__device__ float3 calculateRepulsionForce(
									   float3 const& vel,
									   float3 const& normal,
									   float const& boundary_distance,
									   float const& boundary_dampening,
									   float const& boundary_stiffness
									   )
{

	// from ama06
	return (boundary_stiffness * boundary_distance - boundary_dampening * dot(normal, vel)) * normal;
}


/*
COLLISION RESPONSE
SIMPLE REFLECTION
................
....^...........
Vn..|.../.V.....
....|../........
....|./.........
....|/________>.
........Vt......
................

Vn = (Vbc * n)Vbc
Vt = Vbc –vn
Vbc = velocity before collision
Vn = normal component of velocity
Vt == tangential component of velocity
V = (1-u)Vt – eVn
u = dynamic friction (affects tangent velocity)
e = resilience (affects normal velocity)
*/

__device__ float3 calculateFrictionForce(
									   float3 const& vel,
									   float3 const& force,
									   float3 const& normal,
									   float const& friction_kinetic,
									   float const& friction_static_limit
									   )
{
	float3 friction_force = make_float3(0,0,0);

	// the normal part of the force vector (ie, the part that is going "towards" the boundary
	float3 f_n = force * dot(normal, force);
	// tangent on the terrain along the force direction (unit vector of tangential force)
	float3 f_t = force - f_n;

	// the normal part of the velocity vector (ie, the part that is going "towards" the boundary
	float3 v_n = vel * dot(normal, vel);
	// tangent on the terrain along the velocity direction (unit vector of tangential velocity)
	float3 v_t = vel - v_n;

	if((v_t.x + v_t.y + v_t.z)/3.0f > friction_static_limit)
		friction_force = -v_t;
	else
		friction_force = friction_kinetic * -v_t;

	// above static friction limit?
//  	friction_force.x = f_t.x > friction_static_limit ? friction_kinetic * -v_t.x : -v_t.x;
//  	friction_force.y = f_t.y > friction_static_limit ? friction_kinetic * -v_t.y : -v_t.y;
//  	friction_force.z = f_t.z > friction_static_limit ? friction_kinetic * -v_t.z : -v_t.z;

	//TODO; friction should cause energy/heat in contact particles!
	friction_force = friction_kinetic * -v_t;

	return friction_force;
}

#endif
