/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __TOOLS_STATISTICMANAGER_BASE_H_
#define __TOOLS_STATISTICMANAGER_BASE_H_

#include "baseobject.h"

namespace Tools
{
struct ObjectStats
{
    int created;
    int deleted;
    ObjectStats()
    {
        created = 0;
        deleted = 0;
    };
};

class StatisticManagerBase : public BaseObject
{
protected:
    StatisticManagerBase();
    StatisticManagerBase(string className, int objID);

public:
    virtual ~StatisticManagerBase();

    virtual bool createdObj(string className);
    virtual bool deletedObj(string className);

    virtual void print();
};
};
#endif
