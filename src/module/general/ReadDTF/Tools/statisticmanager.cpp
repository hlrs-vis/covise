/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "statisticmanager.h"

using namespace Tools;

CLASSINFO_OBJ(ClassInfo_ToolsStatisticManager, StatisticManager,
              "Tools::StatisticManager", INT_MAX);

StatisticManager::StatisticManager()
{
    objectStats.clear();
}

StatisticManager::StatisticManager(string className, int objID)
    : StatisticManagerBase(className, objID)
{
    objectStats.clear();
}

StatisticManager::~StatisticManager()
{
    objectStats.clear();
}

bool StatisticManager::createdObj(string className)
{
    map<string, ObjectStats>::iterator statsIterator = this->objectStats.find(className);

    if (statsIterator != this->objectStats.end())
    {
        statsIterator->second.created++;
    }
    else
    {
        ObjectStats stats;

        stats.created++;

        objectStats.insert(pair<string, ObjectStats>(className, stats));
    }

    return true;
}

bool StatisticManager::deletedObj(string className)
{
    map<string, ObjectStats>::iterator statsIterator
        = this->objectStats.find(className);

    if (statsIterator != this->objectStats.end())
    {
        statsIterator->second.deleted++;
        return true;
    }
    else
        return false;
}

void StatisticManager::print()
{
    printf("test print out\n");

    cout << "---------------------------------------------------------" << endl;
    cout << "Statistics: " << endl;
    cout << "---------------------------------------------------------" << endl;

    unsigned int objCreated = 0;
    unsigned int objDeleted = 0;

    int objDelta = 0;

    map<string, ObjectStats>::iterator statsIterator;

    statsIterator = this->objectStats.begin();

    while (statsIterator != this->objectStats.end())
    {
        cout << "class " << statsIterator->first << ": ";
        cout << "\t\tcreated: " << statsIterator->second.created;
        cout << "\tdeleted: " << statsIterator->second.deleted << endl;

        objCreated += statsIterator->second.created;
        objDeleted += statsIterator->second.deleted;

        ++statsIterator;
    }

    objDelta = objCreated - objDeleted;

    if (objDelta == 0)
        cout << "\nHooray: No memory leaks produced by class manager.\n" << endl;
    else
    {
        if (objDelta > 0)
            cout << "\nDamn: Memory leak produced by class manager. More objects "
                 << "created than deleted.\n" << endl;
        else
            cout << "\nDamn: Something went wrong in class manager. More objects "
                 << "deleted than created..\n" << endl;
    }
}
