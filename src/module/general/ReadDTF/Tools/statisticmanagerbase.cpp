/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "statisticmanagerbase.h"

using namespace Tools;

StatisticManagerBase::StatisticManagerBase() {}
StatisticManagerBase::StatisticManagerBase(string className, int objID)
    : BaseObject(className, objID)
{
}

StatisticManagerBase::~StatisticManagerBase() {}

bool StatisticManagerBase::createdObj(string className)
{
    return true;
}

bool StatisticManagerBase::deletedObj(string className)
{
    return true;
}

void StatisticManagerBase::print()
{
}
