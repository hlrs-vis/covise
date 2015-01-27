#ifndef __K_Boundaries_Terrain_cu__
#define __K_Boundaries_Terrain_cu__

#include "K_Boundaries_Common.inl"

#define EPSILON			0.00001f			//for collision detection


__device__ int2 getTerrainPos(float3 const &pos, int const &dTerrainSize, float const &dTerrainWorldSize)
{            
	int2 terrainPos;
	terrainPos.y = floor(pos.z*(dTerrainSize/dTerrainWorldSize));
	terrainPos.x = floor(pos.x*(dTerrainSize/dTerrainWorldSize));
	return terrainPos;
}

__device__ float getTerrainHeight(int const &terrainPosX, int const &terrainPosZ, float const *dTerrainHeights, int const &dTerrainSize)
{            
	return dTerrainHeights[((dTerrainSize) * (dTerrainSize) - 1) - (((dTerrainSize) * terrainPosZ)) + terrainPosX];
}

__device__ float getTerrainHeight(int2 const &terrainPos, float const *dTerrainHeights, int const &dTerrainSize)
{            
	return getTerrainHeight(terrainPos.x, terrainPos.y, dTerrainHeights, dTerrainSize);
}

__device__ float getTerrainHeightInterpolate(
	float3 const &pos, 
	int const &dTerrainSize,
	float const &dTerrainWorldSize, 
	float const *dTerrainHeights)
{            
	int2 tpos = getTerrainPos(pos, dTerrainSize, dTerrainWorldSize);

	int Xa = tpos.x;      // x on one side
	int Xb = tpos.x + 1; // x on the other side
	int Za = tpos.y;      // z on one side
	int Zb = tpos.y+1; // z on the other side

	float Xd = pos.x-floor(pos.x);
	if (Xd < 0.0f)
		Xd *= -1.0f;
	float Zd = pos.z-floor(pos.y);
	if (Zd < 0.0f)
		Zd *= -1.0f;

	float b = lerp(getTerrainHeight(Xa,Zb, dTerrainHeights, dTerrainSize),getTerrainHeight(Xb,Zb, dTerrainHeights, dTerrainSize),Xd);
	float a = lerp(getTerrainHeight(Xa,Za, dTerrainHeights, dTerrainSize),getTerrainHeight(Xb,Za, dTerrainHeights, dTerrainSize),Xd);
	return lerp(a,b,Zd);

}

__device__ float3 getTerrainNormal(
								   int2 const &terrainPos, 
								   float4 const *dTerrainNormals, 
								   int const &dTerrainSize)
{          
	// TODO: perhaps interpolate normals (with curve estimation?)
	float4 normal = (dTerrainNormals[((dTerrainSize) * (dTerrainSize)) - (((dTerrainSize) * terrainPos.y)) + terrainPos.x]);
	return make_float3(normal);
}

__device__ float3 calculateTerrainNoPenetrationForce(
	float3 & pos,
	float3 const& vel,
	float3 const& fluidWorldPosition,
	TerrainData const &dTerrainData,
	float const& boundary_distance,
	float const& boundary_stiffness,
	float const& boundary_dampening,
	float const& scale_to_simulation
	)
{
	float3 repulsion_force = make_float3(0,0,0);
	float diff;

	int2 terrainPos = getTerrainPos(pos+fluidWorldPosition+dTerrainData.position, dTerrainData.size, dTerrainData.world_size);

	if(terrainPos.x >= 0 && terrainPos.x < dTerrainData.size && terrainPos.y >= 0 && terrainPos.y < dTerrainData.size)
	{
		//float terrainHeight = getTerrainHeightInterpolate(pos, dTerrainData.size,dTerrainData.world_size, dTerrainData.heights);
		float terrainHeight = -dTerrainData.position.y - fluidWorldPosition.y + getTerrainHeight(terrainPos, dTerrainData.heights, dTerrainData.size);
		float3 terrainNormal = getTerrainNormal(terrainPos, dTerrainData.normals, dTerrainData.size);
		
		if(pos.y < terrainHeight)
			pos.y  = terrainHeight;

		diff = 2 * boundary_distance - (pos.y  - terrainHeight) * scale_to_simulation;			
		if (diff > EPSILON) 
		{
			repulsion_force += calculateRepulsionForce(vel, terrainNormal, diff, boundary_dampening, boundary_stiffness);
		}
	}
	return repulsion_force;
}


__device__ float3 calculateTerrainFrictionForce(
	float3 const& pos,
	float3 const& vel,
	float3 const& force,
	float3 const& fluidWorldPosition,
	TerrainData const &dTerrainData,
	float const& boundary_distance,
	float const& friction_kinetic,
	float const& friction_static_limit,
	float const& scale_to_simulation
	)
{
	float3 friction_force = make_float3(0,0,0);
	float diff;

	int2 terrainPos = getTerrainPos(pos+fluidWorldPosition+dTerrainData.position, dTerrainData.size, dTerrainData.world_size);

	if(terrainPos.x >= 0 && terrainPos.x < dTerrainData.size && terrainPos.y >= 0 && terrainPos.y < dTerrainData.size )
	{
		//float terrainHeight = getTerrainHeightInterpolate(pos, dTerrainData.size,dTerrainData.world_size, dTerrainData.heights);
		float terrainHeight = -dTerrainData.position.y - fluidWorldPosition.y +getTerrainHeight(terrainPos, dTerrainData.heights, dTerrainData.size);
		float3 terrainNormal = getTerrainNormal(terrainPos, dTerrainData.normals, dTerrainData.size);
		
		// simple limit for terrain collision
		diff = 3 * boundary_distance - (pos.y - terrainHeight) * scale_to_simulation;			
		if (diff > EPSILON) 
		{
			friction_force += calculateFrictionForce(vel, force, terrainNormal, friction_kinetic, friction_static_limit);
		}
	}
	return friction_force;
}

#endif