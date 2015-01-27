/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PLYBNDLNS_H
#define _PLYBNDLNS_H
/**************************************************************************\
**                                   (C)2010 Stellba Hydro GmbH & Co. KG  **
**                                                                        **
** Description:  PolyBoundaryLines                                        **                                                                        **
**                                                                        **
** extracts boundary lines of polygons                                    **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <api/coSimpleModule.h>
#include <do/coDoData.h>

using namespace covise;

class PolyBoundaryLines : public coSimpleModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *paramName, bool inMapLoading);

    //  member data
    coInputPort *p_polygons;
    coInputPort *p_dataIn;

    coOutputPort *p_boundaryLines;
    coOutputPort *p_dataOut;

public:
    PolyBoundaryLines(int argc, char *argv[]);
    virtual ~PolyBoundaryLines();
};

#endif
