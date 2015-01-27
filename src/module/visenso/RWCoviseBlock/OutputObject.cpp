/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OutputObject.h"
#include <iostream>
#include <stdio.h>

OutputObject::OutputObject(const std::string &type)
    : type_(type)
    , distrObj_(NULL)
{
}

OutputObject::OutputObject(const OutputObject &o)
    : type_("NONE")
    , distrObj_(o.distrObj_)
{
    //    std::string x = o.type_;
}

OutputObject::OutputObject()
    : type_("NONE")
    , distrObj_(NULL)
{
}

OutputObject *OutputObject::clone() const
{
    std::cerr << "OutputObject::clone() called  type: " << type_ << std::endl;

    return new OutputObject(*this);
}

std::string OutputObject::type() const
{
    return type_;
}

bool OutputObject::process(const int & /*fd*/)
{
    std::cerr << "process(FILE *fd) NOT IMPLEMENTED YET for OutputObject of type: " << type_ << endl;

    return false;
}

void OutputObject::setDO(const coDistributedObject *d)
{
    distrObj_ = d;
}
