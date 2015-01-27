/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/libobject.cpp
 * @brief contains definition of methods for class DTF_Lib::LibObject
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "libobject.h"

using namespace DTF_Lib;

LibObject::LibObject()
{
    fileHandle = -1;
}

LibObject::LibObject(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
}

LibObject::~LibObject()
{
    clear();
}

bool LibObject::setFileHandle(int handle)
{
    if (handle >= 0)
    {
        this->fileHandle = handle;
        return true;
    }

    return false;
}

bool LibObject::setFileName(string fileName)
{
    this->fileName = fileName;

    return true;
}

int LibObject::getFileHandle()
{
    return this->fileHandle;
}

bool LibObject::implementMe()
{
    cout << "implement me. i don't work. really. believe me." << endl;

    return false;
}

/** EOC */
