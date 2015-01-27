#ifndef __CommonSPH_cu__
#define __CommonSPH_cu__

enum SPHSymmetrization
{
	SPH_PRESSURE_MUELLER = 0,
	SPH_PRESSURE_VISCOPLASTIC = 1
};

enum SPHColoringSource
{
	Pressure,
	Velocity,
	Force
};


#endif