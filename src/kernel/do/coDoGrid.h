/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_GRID_H
#define CO_GRID_H

#include "coDistributedObject.h"

namespace covise
{

class DOEXPORT coDoGrid : public coDistributedObject
{
public:
    explicit coDoGrid(const coObjInfo &info)
        : coDistributedObject(info)
    {
    }
    explicit coDoGrid(const coObjInfo &info, const char *t)
        : coDistributedObject(info, t)
    {
    }
    virtual int getNumPoints() const = 0;
};
}
#endif
