#ifndef __K_SimpleSPH_cu__
#define __K_SimpleSPH_cu__

#include "K_Common.cuh"
#include "cutil_math.h"
#include "vector_types.h"

using namespace SimLib;
using namespace SimLib::Sim::SimpleSPH;

#include "K_UniformGrid_Utils.inl"
#include "K_Coloring.inl"
#include "K_SPH_Common.cuh"

class SimpleSPHSystem
{
public:

   static __device__ void UpdateSortedValues(SimpleSPHData &dParticlesSorted, SimpleSPHData &dParticles, uint &index, uint &sortedIndex)
   {
      dParticlesSorted.position[index] = FETCH_NOTEX(dParticles,position,sortedIndex);
      dParticlesSorted.velocity[index] = FETCH_NOTEX(dParticles,velocity,sortedIndex);
      dParticlesSorted.veleval[index]     = FETCH_NOTEX(dParticles,veleval,sortedIndex);
      //dParticlesSorted.color[index]     = FETCH_NOTEX(dParticles,color,sortedIndex);

      //dParticlesSorted.veleval_diff[index]= FETCH_NOTEX(dParticles,veleval_diff,sortedIndex);
      //dParticlesSorted.sph_force[index] = FETCH_NOTEX(dParticles,sph_force,sortedIndex);
      //dParticlesSorted.pressure[index]  = FETCH_NOTEX(dParticles,pressure,sortedIndex);
      //dParticlesSorted.density[index]      = FETCH_NOTEX(dParticles,density,sortedIndex);
   }
};


#include "K_SimpleSPH_Step1.inl"
#include "K_SimpleSPH_Step2.inl"
#include "K_SimpleSPH_Integrate.inl"


#endif
