/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/info.cpp
 * @brief contains definition of methods for class DTF_Lib::Info
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "info.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibInfo, Info, "DTF_Lib::Info", 1);

Info::Info()
    : LibObject()
{
}

Info::Info(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

Info::~Info()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Info::queryApplication(string &application)
{
    return implementMe();
}

bool Info::queryAppVersion(string &version)
{
    return implementMe();
}

bool Info::queryCreateTime(time_t &created)
{
    int handle = this->fileHandle;

    if ((created = dtf_query_cretime(&handle)) != DTF_ERROR)
        return true;

    return false;
}

bool Info::queryDtfVersion(string &version)
{
    return implementMe();
}

bool Info::queryFileVersion(string &version)
{
    dtf_string fileVersion;
    int handle = this->fileHandle;

    if (dtf_query_file_version(&handle, fileVersion) != DTF_ERROR)
    {
        version = fileVersion;

        return true;
    }

    return false;
}

bool Info::queryModTime(time_t &modified)
{
    int handle = this->fileHandle;

    if ((modified = dtf_query_modtime(&handle)) != DTF_ERROR)
        return true;

    return false;
}

bool Info::queryOrigin(string &origin)
{
    dtf_string fileOrigin;
    int handle = this->fileHandle;

    if (dtf_query_origin(&handle, fileOrigin) != DTF_ERROR)
    {
        origin = fileOrigin;

        return true;
    }

    return false;
}

bool Info::queryScaling(double &scaling)
{
    int handle = this->fileHandle;

    if ((scaling = dtf_query_scaling_d(&handle)) != DTF_ERROR)
        return true;

    return false;
}

bool Info::queryTitle(string &title)
{
    int status;
    dtf_string fileTitle;
    int handle = this->fileHandle;

    status = dtf_query_title(&handle, fileTitle);

    if (status != DTF_ERROR)
    {
        title = fileTitle;

        return true;
    }

    return false;
}

/** EOC */
