/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PROSI_READMP_H
#define PROSI_READMP_H

/************************************************************************
 *                                                                      *
 *                                                                      *
 *     High Performance Computer Centre University of Stuttgart	        *
 *                         Allmandring 30a                              *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                    Reader for measurement reports                    *
 *									*
 ************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <list>

class ReadMP : public coModule
{

public:
    ReadMP();
    virtual ~ReadMP();

    virtual int compute();

private:
    coDoPoints *points;
    coDoLines *lines;

    coOutputPort *mpPointsPort;
    coOutputPort *mpLinesPort;

    coFileBrowserParam *mpFileParam;

    void makeGeos(const std::list<float> &vertices, coDoPoints *points, coDoLines *lines, long limit = -1);
};
#endif
