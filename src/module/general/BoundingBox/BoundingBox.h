/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ENLARGE_H
#define _ENLARGE_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: Filter program in COVISE API                           ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <do/coDoLines.h>

class BoundingBox : public coModule
{
private:
    //////////  member functions
    virtual int compute(const char *port);
    void calcBB(coDistributedObject *obj, float min[3], float max[3]);

    /// creates the outer lines of a box with the origin(ox,oy,oz) and
    /// the sizes (size_x, size_y, size_z)
    coDoLines *createBox(const char *objectName, float ox, float oy, float oz,
                         float size_x, float size_y, float size_z);

    ////////// ports
    coInputPort *p_mesh;
    coOutputPort *p_box;

public:
    BoundingBox(int argc, char *argv[]);
};
#endif
