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

void actionClass::addMember(const char *name, const char *objectName)
{
    DxObjectMap::iterator cur = objects_.find(objectName);
    if (cur == objects_.end())
    {
        printf("There is no object named \"%s\" required for \"%s\"\n", objectName, name);
        /*
      printf("objects:\n");
      for (cur = objects_.begin(); cur != objects_.end(); cur ++) {
         printf(" [%s]", cur->first);
      }
      printf("\n");
*/
    }
    else
        currObject_->addMember(new MultiGridMember(name, objects_[objectName]));
}

void actionClass::addMember(int number, const char *objectName)
{
    DxObjectMap::iterator cur = objects_.find(objectName);
    if (cur == objects_.end())
    {
        printf("There is no object named \"%s\" required for \"%d\"\n", objectName, number);
        /*
      printf("objects:\n");
      for (cur = objects_.begin(); cur != objects_.end(); cur ++) {
         printf(" [%s]", cur->first);
      }
      printf("\n");
*/
    }
    else
        currObject_->addMember(new MultiGridMember(number, objects_[objectName]));
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
