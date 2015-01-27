/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __TOOLS_STATISTICMANAGER_H_
#define __TOOLS_STATISTICMANAGER_H_

#include "classmanager.h"

#include <cstdio>
using namespace std;

namespace Tools
{
class ClassInfo_ToolsStatisticManager;

class StatisticManager : public StatisticManagerBase
{
    friend class ClassInfo_ToolsStatisticManager;

private:
    map<string, ObjectStats> objectStats;

    StatisticManager();
    StatisticManager(string className, int objID);

    static ClassInfo_ToolsStatisticManager classInfo;

public:
    virtual ~StatisticManager();

    bool createdObj(string className);
    bool deletedObj(string className);

    virtual void print();
};

CLASSINFO(ClassInfo_ToolsStatisticManager, StatisticManager);
};
#endif
