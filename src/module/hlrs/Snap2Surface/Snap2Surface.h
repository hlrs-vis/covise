#ifndef SNAP2SURFACE_H
#define SNAP2SURFACE_H

#include <api/coSimpleModule.h>
#include <do/covise_gridmethods.h>


using namespace covise;


class Snap2Surface : public coSimpleModule
{
private:
    coInputPort *p_pointsIn;
    coInputPort *p_surfaceIn;

    coOutputPort *p_pointsOut;

    coChoiceParam *p_axis;
    coFloatParam *p_delta;

    virtual int compute(const char *port);

    bool outOfBox(grid_methods::POINT3D pt , int ax, grid_methods::POINT3D max, grid_methods::POINT3D min);
    bool outOfQuad(int* idx, grid_methods::POINT3D pt, int ax);
    float *x_surf,*y_surf,*z_surf;

public:
  Snap2Surface(int argc, char *argv[]);
};

#endif
