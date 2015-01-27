/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MultiGridMember.h"
MultiGridMember::MultiGridMember(const char *name, DxObject *object)
{
    name_ = new char[1 + strlen(name)];
    strcpy(name_, name);
    object_ = object;
}

MultiGridMember::MultiGridMember(int number, DxObject *object)
{
    name_ = new char[100];
    sprintf(name_, "%d", number);
    object_ = object;
}

MultiGridMember::~MultiGridMember()
{
    delete name_;
    //object MUST NOT be deleted
}
