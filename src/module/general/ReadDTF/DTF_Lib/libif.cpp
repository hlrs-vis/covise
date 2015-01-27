/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/libif.cpp
 * @brief contains definition of methods for class DTF_Lib::LibIF
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "libif.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibLibIF, LibIF, "DTF_Lib::LibIF", 1);

LibIF::LibIF(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

LibIF::~LibIF()
{
    clear();

    DEC_OBJ_COUNT(getClassName());
}

LibIF::LibIF()
{
    fileName = "";
}

bool LibIF::setFileName(string fileName)
{
    LibObject *file = (LibObject *)getChildFunc("File");
    map<string, Tools::BaseObject *>::iterator funcIterator;

    if (file != NULL)
        if (file->setFileName(fileName))
        {
            funcIterator = this->childFuncs.begin();

            while (funcIterator != childFuncs.end())
            {
                LibObject *obj = (LibObject *)funcIterator->second;

                if (funcIterator->first != "File")
                {
                    obj->setFileName(fileName);
                    obj->setFileHandle(file->getFileHandle());
                }

                ++funcIterator;
            }

            return true;
        }

    return false;
}

Tools::BaseObject *LibIF::operator()(string className)
{
    return getChildFunc(className);
}

bool LibIF::init()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    // fill childFuncs
    addChildFunc("File", cm->getObject("DTF_Lib::File"));
    addChildFunc("Info", cm->getObject("DTF_Lib::Info"));
    addChildFunc("Misc", cm->getObject("DTF_Lib::Misc"));
    addChildFunc("Sim", cm->getObject("DTF_Lib::Sim"));
    addChildFunc("Zone", cm->getObject("DTF_Lib::Zone"));

    return true;
}

/** EOC */
