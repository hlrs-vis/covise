/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "ResultParam.h"
#include <stdio.h>

ResultParam::ResultParam(Type ptype)
{
    dirname_ = NULL;
    name_ = NULL;
    val_ = NULL;
    ptype_ = ptype;
}

ResultParam::~ResultParam()
{
    delete[] dirname_;
    delete[] name_;
    delete[] val_;
}

void
ResultParam::setLabel(const char *name, const char *value)
{
    delete[] name_;
    name_ = new char[strlen(name) + 1];
    strcpy(name_, name);

    setLabel(value);
}

void
ResultParam::setLabel(const char *value)
{
    delete[] val_;
    val_ = new char[strlen(value) + 1];
    strcpy(val_, value);
}

const char *
ResultParam::getDirName()
{
    delete[] dirname_;

    dirname_ = new char[strlen(name_) + strlen(val_) + 2];
    sprintf(dirname_, "%s=%s", name_, val_);
    return dirname_;
}
