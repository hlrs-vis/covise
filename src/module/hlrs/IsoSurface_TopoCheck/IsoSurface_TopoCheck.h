/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ISOTOPO_H
#define _ISOTOPO_H
/****************************************************************************\ 
 **                                                                          **
 ** Description: IsoSurface for UniformGrids                                 **
 **              uses Thomas Lewiner's algorithm that checks                 **
 **              for topologically correct results                           **
 **                                                                          **
 ** Name:        IsoSurface_TopoCheck                                        **
 ** Category:    Mapper                                                      **
 **                                                                          **
 ** Author: M. Becker                                                        **
 ** 01/2009                                                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include "marching_cubes_jgt/MarchingCubes.h"

class IsoSurface_TopoCheck : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    virtual void postInst();

    //  Ports
    coInputPort *p_gridIn;
    coInputPort *p_dataIn;

    coOutputPort *p_polyOut;
    coOutputPort *p_normalsOut;

    coFloatParam *p_isovalue;

public:
    IsoSurface_TopoCheck(int argc, char *argv[]);
    virtual ~IsoSurface_TopoCheck();
};
#endif
