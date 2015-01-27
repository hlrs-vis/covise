/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "action.h"
#include <util/coviseCompat.h>
#include <api/coModule.h>

using namespace covise;

actionClass::actionClass(int defaultByteOrder)
{
    // arrays_ = new map <const char *, DxObject *, ltstr> ;
    defaultByteOrder_ = defaultByteOrder;
    currObject_ = new DxObject(defaultByteOrder);
}

void actionClass::addMember(const char *name, const char *fieldName)
{
    DxObjectMap::iterator cur = fields_.find(fieldName);
    if (cur == fields_.end())
    {
        char msg[1024];
        sprintf(msg, "There is no fieldobject named \"%s\" required for \"%s\"", fieldName, name);
        Covise::sendError(msg);
    }
    else
    {
        multiGridMembers_.push_back(new MultiGridMember(name, fields_[fieldName]));
    }
}

void actionClass::addMember(int number, const char *fieldName)
{
    DxObjectMap::iterator cur = fields_.find(fieldName);
    if (cur == fields_.end())
    {
        char msg[1024];
        sprintf(msg, "There is no fieldobject named \"%s\" required for \"%d\"", fieldName, number);
        Covise::sendError(msg);
    }
    else
    {
        multiGridMembers_.push_back(new MultiGridMember(number, fields_[fieldName]));
    }
}

//void actionClass::setCurrFilename(const char *filename)
void actionClass::setCurrFileName(const char *filename)
{
    currObject_->setFileName(filename);
}

void actionClass::setCurrFileName(const char *dirname, const char *filename)
{
    char *path = new char[1 + strlen(dirname) + strlen(filename)];
    strcpy(path, dirname);
    if ('/' != dirname[strlen(dirname) - 1])
    {
        strcat(path, "/");
    }
    strcat(path, filename);
    currObject_->setFileName(path);
    delete path;
}

// stringvalue {$$ = new attribute($1->getString());}
void actionClass::setCurrName(const char *name)
{
    currObject_->setName(name);
}

void actionClass::setCurrName(int number)
{
    currObject_->setName(number);
}

void actionClass::setCurrElementType(const char *elementType)
{
    currObject_->setElementType(elementType);
}

void actionClass::setCurrRef(const char *ref)
{
    currObject_->setRef(ref);
}

void actionClass::setCurrData(const char *data)
{
    currObject_->setData(data);
}

void actionClass::setCurrConnections(const char *connections)
{
    currObject_->setConnections(connections);
}

void actionClass::setCurrPositions(const char *positions)
{
    currObject_->setPositions(positions);
}

void actionClass::setCurrData(int data)
{
    currObject_->setData(data);
}

void actionClass::setCurrConnections(int connections)
{
    currObject_->setConnections(connections);
}

void actionClass::setCurrPositions(int positions)
{
    currObject_->setPositions(positions);
}

void actionClass::setCurrAttributeDep(const char *attributeDep)
{
    currObject_->addAttributeDep(attributeDep);
}

void actionClass::setCurrAttributeName(const char *attributeName)
{
    currObject_->addAttributeName(attributeName);
}

void actionClass::show()
{
    currObject_->show();
}
