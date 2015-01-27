/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file classinfo.cpp
 * @brief contains definition of methods of class ClassInfo.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 12.10.2003
 * created
 */

#include "classinfo.h"

using namespace Tools;

ClassInfo::ClassInfo(string className, int maxObj)
{
#ifdef DEBUG_MODE
    cout << "ClassInfo::ClassInfo(): " << className << endl;
#endif

    this->className = className;

    this->maxObj = maxObj;
}

ClassInfo::~ClassInfo()
{
#ifdef DEBUG_MODE
    cout << "ClassInfo::~ClassInfo(): " << className << endl;
#endif
}

BaseObject *ClassInfo::New(int objectID)
{
    return new BaseObject(this->className, objectID);
}

bool ClassInfo::maxObjReached()
{
    return false;
}

int ClassInfo::getMaxObj()
{
    return maxObj;
}

int ClassInfo::getNumObj()
{
    return -1;
}

string ClassInfo::getClassName()
{
    return this->className;
}
