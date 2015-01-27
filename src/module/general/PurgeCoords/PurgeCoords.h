/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PURGECOORDS_H
#define _PURGECOORDS_H

#include <api/coSimpleModule.h>
using namespace covise;

#include <do/coDoPolygons.h>
#include <do/coDoUnstructuredGrid.h>

class PurgeCoords : public coSimpleModule
{
public:
private:
    //////////  member functions
    virtual int compute(const char *port);
    virtual void quit();
    virtual void postInst();

    ////////// the data in- and output ports
    coInputPort *p_GridIn, *p_ScalDataIn, *p_VecDataIn;
    coOutputPort *p_GridOut, *p_ScalDataOut, *p_VecDataOut;

public:
    PurgeCoords(int argc, char *argv[]);
};
#endif // _PURGECOORDS_H
