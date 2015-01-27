#ifndef __K_SPH_Kernels_cu__
#define __K_SPH_Kernels_cu__

#ifndef M_PI
#define M_PI	3.14159265358979323846f
#endif

#ifndef M_1_PI 
#define M_1_PI	0.31830988618379067154f
#endif

namespace SPH_Kernels
{
	#include "K_SPH_Kernels_gaussian.inl"
	#include "K_SPH_Kernels_quintic.inl"
	#include "K_SPH_Kernels_quartic.inl"
	#include "K_SPH_Kernels_quadratic.inl"
	#include "K_SPH_Kernels_cubic.inl"
	#include "K_SPH_Kernels_poly6.inl"
	#include "K_SPH_Kernels_spiky.inl"
	#include "K_SPH_Kernels_viscosity.inl"
}
#endif