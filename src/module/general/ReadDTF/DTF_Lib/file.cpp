/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/file.cpp
 * @brief contains definition of methods for class DTF_Lib::File
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include <stdio.h>
#include "file.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibFile, File, "DTF_Lib::File", 1);

File::File()
    : LibObject()
{
}

File::File(string className, int objectID)
    : LibObject(className, objectID)
{
    this->fileHandle = -1;
    INC_OBJ_COUNT(getClassName());
}

File::~File()
{
    close();

    DEC_OBJ_COUNT(getClassName());
}

bool File::setFileName(string fileName)
{
    close();
    if (!this->open(fileName))
        return false;

    return true;
}

int File::getFileHandle()
{
    return this->fileHandle;
}

bool File::open(string fileName)
{
    int status = dtf_open_file(fileName.c_str());

    if (status != DTF_ERROR)
    {
        this->fileHandle = status;

        return true;
    }
    else
        this->fileHandle = -1;

    return false;
}

bool File::close()
{
    int status = DTF_ERROR;

    if (this->fileHandle > -1)
    {
        status = dtf_close_file(&(this->fileHandle));
        this->fileHandle = -1;
    }

    if (status == DTF_ERROR)
        return false;

    return true;
}

bool File::queryNumSims(int &numSims)
{
    int handle = this->fileHandle;

    numSims = dtf_query_nsims(&handle);

    if ((numSims != DTF_ERROR) && (numSims >= 0))
        return true;

    return false;
}

/** EOC */
