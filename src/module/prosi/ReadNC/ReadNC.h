/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PROSI_READNC_H
#define PROSI_READNC_H

/************************************************************************
 *                                                                      *
 *                                                                      *
 *     High Performance Computer Centre University of Stuttgart	        *
 *                         Allmandring 30a                              *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                    Reader for NC machine data                        *
 *									*
 ************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <list>

class ReadNC : public coModule
{

public:
    ReadNC();
    virtual ~ReadNC();

    virtual int compute();

private:
    coDoPoints *points;
    coDoLines *lines;

    coDoSet *pointSet;
    coDoSet *lineSet;

    coDoVec3 *orientations;
    coDoFloat *angles;
    coDoFloat *feeds;
    coDoFloat *spindlespeeds;
    coDoFloat *gfunctions;
    coDoFloat *forces;

    coOutputPort *ncPointsPort;
    coOutputPort *ncLinesPort;
    coOutputPort *ncOrientationPort;
    coOutputPort *ncOrientationAnglePort;
    coOutputPort *ncFeedPort;
    coOutputPort *ncSpindleSpeedPort;
    coOutputPort *ncGFunctionPort;
    coOutputPort *ncForcesPort;

    coFileBrowserParam *ncFileParam;
    coBooleanParam *ncTimestepParam;
    coIntScalarParam *ncNoTimestepsParam;

    void makeGeos(const std::list<float> &vertices, coDoPoints *points, coDoLines *lines, long limit = -1);
};
#endif
