/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COMPUTEGRID_H
#define COMPUTEGRID_H

// includes
#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <util/coRestraint.h>
#include <do/coDoSet.h>
#include <do/coDoStructuredGrid.h>

class ComputeGrid : public coSimpleModule
{
public:
    ComputeGrid(int argc, char *argv[]);
    int compute(const char *port);
    

private:
    //ports
    coInputPort *p_pointsIn;
    coOutputPort *p_pointsOut;
    coOutputPort *p_GridOut;
    
    //parameters
    coIntSliderParam *g_x_dim, *g_y_dim, *g_z_dim;
    coStringParam *p_particle, *pp_particle;
    coIntScalarParam *p_maxParticleNumber;
    coFloatVectorParam *p_boundingBoxDimensions;
    coDoPoints *getSelectedPointCoordinates(const std::string &name, const coDistributedObject *points);
    
    vector<ssize_t> m_particleSelection;
    coRestraint m_particleIndices;

};
#endif
