/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/File.h
 * @brief contains implementation of class DTF::File
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 */

#include "File.h"

using namespace std;
using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFFile, File, "DTF::File", 1);

File::File()
{
    INC_OBJ_COUNT(getClassName());
}

File::File(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    title = "none";
    origin = "none";
    dtfVersion = "";
    scaling = 1.0;

    INC_OBJ_COUNT(getClassName());
}

File::~File()
{
    DEC_OBJ_COUNT(getClassName());
}

string File::getTitle()
{
    return this->title;
}

string File::getOrigin()
{
    return this->origin;
}

bool File::getCreationTime(time_t &creationTime)
{
    creationTime = this->created;

    return true;
}

bool File::getModTime(time_t &modTime)
{
    modTime = this->modified;

    return true;
}

string File::getDTFVersion()
{
    return this->dtfVersion;
}

double File::getScaling()
{
    return this->scaling;
}

bool File::read(string fileName)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DTF_Lib::LibIF *libif = (DTF_Lib::LibIF *)cm->getObject("DTF_Lib::LibIF");
    DTF_Lib::Info *info = (DTF_Lib::Info *)(*libif)("Info");

    string store = "";
    double scalingValue;

    if (info->queryTitle(store))
        this->title = store;
    if (info->queryOrigin(store))
        this->origin = store;
    info->queryCreateTime(this->created);
    info->queryModTime(this->modified);
    if (info->queryFileVersion(store))
        this->dtfVersion = store;
    if (info->queryScaling(scalingValue))
        this->scaling = scalingValue;

    return true;
}

void File::print()
{
    time_t timeValue;

    cout << "file info: " << endl;
    cout << "---------------------------------------------------------" << endl;

    cout << "title                      : " << this->getTitle() << endl;
    cout << "origin                     : " << this->getOrigin() << endl;

    if (this->getCreationTime(timeValue))
        cout << "created on                 : " << ctime(&timeValue);

    if (this->getModTime(timeValue))
        cout << "modified on                : " << ctime(&timeValue);

    cout << "DTF version (file)         : " << this->getDTFVersion() << endl;
    cout << "scaling                    : " << this->getScaling() << endl;
}
