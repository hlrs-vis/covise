/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__SPHERE_H)
#define __SPHERE_H

// includes
#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

class Sphere : public coSimpleModule
{
    COMODULE

private:
    //parameters
    coInputPort *p_pointsIn, *p_dataIn, *p_colorsIn;
    coOutputPort *p_spheresOut, *p_normalsOut, *p_dataOut;

    coFloatParam *p_scale, *p_radius;
    coChoiceParam *m_pRenderMethod;

public:
    Sphere(int argc, char **argv);

    int compute(const char *port);
};
#endif
// __SPHERE_H
