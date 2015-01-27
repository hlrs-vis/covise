/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PROSI_GENLINENORMALS_H
#define PROSI_GENLINENORMALS_H

/************************************************************************
 *                                                                      *
 *                                                                      *
 *     High Performance Computer Centre University of Stuttgart	        *
 *                         Allmandring 30a                              *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                    Tries to build line normals                       *
 *									*
 ************************************************************************/

#include <api/coModule.h>
using namespace covise;

class GenLineNormals : public coModule
{

public:
    GenLineNormals();
    virtual ~GenLineNormals();

    virtual int compute();

private:
    coInputPort *linesInPort;
    coOutputPort *normalsOutPort;
};
#endif
