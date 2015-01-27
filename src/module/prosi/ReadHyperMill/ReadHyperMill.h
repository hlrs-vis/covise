/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PROSI_READHYPERMILL_H
#define PROSI_READHYPERMILL_H

/************************************************************************
 *                                                                      *
 *                                                                      *
 *     High Performance Computer Centre University of Stuttgart	        *
 *                         Allmandring 30a                              *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                    Reader for HyperMill data                         *
 *									*
 ************************************************************************/

#include <api/coModule.h>
using namespace covise;

class ReadHyperMill : public coModule
{

public:
    ReadHyperMill();
    virtual ~ReadHyperMill();

    virtual int compute();

private:
    coDoPoints *points;
    coDoLines *lines;

    coDoVec3 *orientations;
    coDoFloat *angles;
    coDoFloat *feeds;

    coOutputPort *hmPointsPort;
    coOutputPort *hmLinesPort;
    coOutputPort *hmOrientationPort;
    coOutputPort *hmOrientationAnglePort;
    coOutputPort *hmFeedPort;

    coFileBrowserParam *hmFileParam;
};
#endif
