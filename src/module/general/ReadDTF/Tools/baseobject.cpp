/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file baseobject.cpp
 * @brief contains implementation of methods for class Tools::BaseObject.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 12.10.2003
 * created
 */
#include "baseobject.h"

using namespace Tools;

int BaseObject::numObj = 0;

BaseObject::BaseObject()
{
    this->objectID = -1;
    BaseObject::numObj++;
}

BaseObject::BaseObject(string className, int objectID)
{
    this->objectID = objectID;
    BaseObject::numObj++;
    clear();

    this->className = className;
}

BaseObject::~BaseObject()
{
    BaseObject::numObj--;
    clear();
}

int BaseObject::getID()
{
    return this->objectID;
}

int BaseObject::getNumObj()
{
    return BaseObject::numObj;
}

bool BaseObject::init()
{
    return true;
}

bool BaseObject::addChildFunc(string shortName, BaseObject *object)
{
    map<string, BaseObject *>::iterator funcIterator = childFuncs.find(shortName);

    if (funcIterator == childFuncs.end())
        childFuncs.insert(pair<string, BaseObject *>(shortName, object));

    return true;
}

BaseObject *BaseObject::getChildFunc(string shortName)
{
    BaseObject *bObj = NULL;

    map<string, BaseObject *>::iterator funcIterator = childFuncs.find(shortName);

    if (funcIterator != childFuncs.end())
        bObj = funcIterator->second;

    return bObj;
}

void BaseObject::clear()
{
    childFuncs.clear();
}

string BaseObject::getClassName()
{
    return this->className;
}
